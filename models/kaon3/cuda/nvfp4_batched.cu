// 解码形状专用：把 top-8 专家并成一次 GEMM（每行一组权重），并测不同 warp 划分。
// 目标：M=1~32 时逼近权重读取带宽（2816x1408 fp4 = 1.98 MB/矩阵）。
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
__host__ __device__ inline float e2m1_f(unsigned n){
  static const float t[8]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  float m=t[n&7u]; return (n&8u)?-m:m; }
__device__ __forceinline__ void mma_fp4(const uint32_t* a,const uint32_t* b,float* d){
  uint16_t z=0; uint32_t s=0x38383838u;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "r"(s),"h"(z),"h"(z),"r"(s),"h"(z),"h"(z));
}
// 宽 N：每 warp 沿 N 铺 32 列（4 个 n8），M 固定 16 行，K 流式。
// block = 8 warp = 4(M) x 2(N)，覆盖 64 x 64
constexpr int BM=64, BN=64, BK=256;
__global__ void k(const uint8_t* __restrict__ A,const uint8_t* __restrict__ B,
                  float* __restrict__ D,int M,int N,int K){
  __shared__ uint8_t sA[BM*BK/2];
  __shared__ uint8_t sB[BN*BK/2];
  const int tid=threadIdx.x, lane=tid%32, warp=tid/32;
  const int wm=warp/4, wn=warp%4;
  const int m_blk=blockIdx.y*BM, n_blk=blockIdx.x*BN;
  float acc[4][4]={};
  const int rb=BK/32, nk=K/BK;
  for(int kb=0;kb<nk;kb++){
    {
      const uint4* src=reinterpret_cast<const uint4*>(A+(size_t)m_blk*(K/2)+kb*(BK/2));
      uint4* dst=reinterpret_cast<uint4*>(sA);
      for(int i=tid;i<(BM*BK/2)/16;i+=blockDim.x) dst[i]=src[(size_t)(i/rb)*(K/32)+(i%rb)];
    }
    {
      const uint4* src=reinterpret_cast<const uint4*>(B+(size_t)n_blk*(K/2)+kb*(BK/2));
      uint4* dst=reinterpret_cast<uint4*>(sB);
      for(int i=tid;i<(BN*BK/2)/16;i+=blockDim.x) dst[i]=src[(size_t)(i/rb)*(K/32)+(i%rb)];
    }
    __syncthreads();
    for(int b=0;b<BK/64;b++){
      const int kb0=b*64, mrow=wm*16, ncol=wn*16;
      uint32_t a[4];
      #pragma unroll
      for(int g=0;g<4;g++){
        int m=(lane/4)+8*(g%2);
        int kb_=kb0+8*(lane%4)+32*(g/2);
        a[g]=*reinterpret_cast<const uint32_t*>(sA+(mrow+m)*(BK/2)+kb_/2);
      }
      #pragma unroll
      for(int q=0;q<2;q++){
        uint32_t bb[2];
        #pragma unroll
        for(int g=0;g<2;g++){
          int kb_=kb0+8*(lane%4)+32*g;
          bb[g]=*reinterpret_cast<const uint32_t*>(sB+(ncol+q*8+(lane/4))*(BK/2)+kb_/2);
        }
        mma_fp4(a,bb,acc[q]);
      }
    }
    __syncthreads();
  }
  int r0=lane/4, cg=(lane%4)*2;
  #pragma unroll
  for(int q=0;q<2;q++){
    int g0=m_blk+wm*16+r0, g1=g0+8, gn=n_blk+wn*16+q*8+cg;
    if(g0<M&&gn<N){D[(size_t)g0*N+gn]=acc[q][0]; if(gn+1<N)D[(size_t)g0*N+gn+1]=acc[q][1];}
    if(g1<M&&gn<N){D[(size_t)g1*N+gn]=acc[q][2]; if(gn+1<N)D[(size_t)g1*N+gn+1]=acc[q][3];}
  }
}
int main(int argc,char**argv){
  int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]),iters=argc>4?atoi(argv[4]):200;
  std::vector<uint8_t>A((size_t)M*K/2),B((size_t)N*K/2);
  for(size_t i=0;i<A.size();i++)A[i]=(i*37)&0xFF;
  for(size_t i=0;i<B.size();i++)B[i]=(i*53)&0xFF;
  uint8_t *dA,*dB;float* dD;
  cudaMalloc(&dA,A.size());cudaMalloc(&dB,B.size());cudaMalloc(&dD,(size_t)M*N*4);
  cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice);
  dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
  k<<<grid,256>>>(dA,dB,dD,M,N,K);
  auto e=cudaDeviceSynchronize();
  if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}
  if(M<=64&&N<=128){
    std::vector<float>D((size_t)M*N);cudaMemcpy(D.data(),dD,(size_t)M*N*4,cudaMemcpyDeviceToHost);
    auto uns=[&](const std::vector<uint8_t>&X,int i){unsigned b=X[i/2];unsigned n=(i%2)?((b>>4)&15):(b&15);return e2m1_f(n);};
    int bad=0;double mx=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int kk=0;kk<K;kk++)s+=(double)uns(A,m*K+kk)*uns(B,n*K+kk);
      double d=fabs(s-D[(size_t)m*N+n]); if(d>1e-2*fmax(1.0,fabs(s)))bad++; mx=fmax(mx,d);}
    printf("PARITY bad=%d/%d maxabs=%.5f\n",bad,M*N,mx);
  }
  cudaEvent_t s0,s1;cudaEventCreate(&s0);cudaEventCreate(&s1);
  cudaEventRecord(s0);
  for(int i=0;i<iters;i++)k<<<grid,256>>>(dA,dB,dD,M,N,K);
  cudaEventRecord(s1);cudaEventSynchronize(s1);
  float ms=0;cudaEventElapsedTime(&ms,s0,s1);ms/=iters;
  double bytes=(double)M*K/2*(N/BN)+(double)N*K/2*(M/BM);
  printf("M=%d N=%d K=%d | %.4f ms | %.1f TF/s | 读 %.0f GB/s (%.2f MB 权重)\n",
         M,N,K,ms,2.0*M*N*K/(ms*1e-3)/1e12,bytes/(ms*1e-3)/1e9,(double)N*K/2/1e6);
  return 0;
}
