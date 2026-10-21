// FP8 GEMM v2：ldmatrix.x4/x2 + smem padding + cp.async 双缓冲 + 8 个累加链。
// block 64x128x64，8 warp = 2(M) x 4(N)，每 warp 32x32（2x4 个 mma tile）。
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__host__ __device__ inline float e4m3_f(unsigned b){
  unsigned s=(b>>7)&1u,e=(b>>3)&15u,m=b&7u;
  if(e==0){float v=m*(1.f/8.f)*powf(2.f,-6.f);return s?-v:v;}
  if(e==15&&m==7) return s?-448.f:448.f;
  float v=(1.f+m*(1.f/8.f))*powf(2.f,(int)e-7);
  return s?-v:v;
}
__device__ __forceinline__ uint32_t smem_u32(const void* p){
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t (&r)[2], uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(r[0]),"=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma_fp8(const uint32_t* a,const uint32_t* b,float* d){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void cp16(uint32_t s,const void* g,int bytes){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s),"l"(g),"r"(bytes));
}

constexpr int BM=64, BN=128, BK=64, PAD=16, RS=BK+PAD;

__global__ void __launch_bounds__(256)
gemm_v2(const uint8_t* __restrict__ A,const uint8_t* __restrict__ B,
        float* __restrict__ D,int M,int N,int K){
  __shared__ uint8_t sA[2][BM*RS];
  __shared__ uint8_t sB[2][BN*RS];
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  const int wm=warp>>2, wn=warp&3;
  const int m_blk=blockIdx.y*BM, n_blk=blockIdx.x*BN;
  float acc[2][4][4]={};

  auto load=[&](int st,int kb){
    { int row=tid>>2, ch=tid&3;
      int gm=m_blk+row;
      const uint8_t* g=A+(size_t)(gm<M?gm:M-1)*K+kb*BK+ch*16;
      cp16(smem_u32(&sA[st][row*RS+ch*16]),g,gm<M?16:0); }
    #pragma unroll
    for(int i=0;i<2;i++){
      int u=tid+i*256, row=u>>2, ch=u&3;
      int gn=n_blk+row;
      const uint8_t* g=B+(size_t)(gn<N?gn:N-1)*K+kb*BK+ch*16;
      cp16(smem_u32(&sB[st][row*RS+ch*16]),g,gn<N?16:0);
    }
    asm volatile("cp.async.commit_group;\n");
  };

  const int nk=K/BK;
  load(0,0);
  for(int kb=0;kb<nk;kb++){
    const int cur=kb&1, nxt=cur^1;
    if(kb+1<nk) load(nxt,kb+1); else asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 1;\n");
    __syncthreads();
    #pragma unroll
    for(int ks=0;ks<2;ks++){
      const int kb0=ks*32;
      uint32_t a[2][4];
      #pragma unroll
      for(int mt=0;mt<2;mt++){
        int r=lane&15, c=(lane>=16)?16:0;
        ldm_x4(a[mt], smem_u32(&sA[cur][(wm*32+mt*16+r)*RS+kb0+c]));
      }
      uint32_t b[4][2];
      #pragma unroll
      for(int nt=0;nt<4;nt++){
        int rb=lane&7, cb=(lane&8)?16:0;
        ldm_x2(b[nt], smem_u32(&sB[cur][(wn*32+nt*8+rb)*RS+kb0+cb]));
      }
      #pragma unroll
      for(int mt=0;mt<2;mt++)
        #pragma unroll
        for(int nt=0;nt<4;nt++)
          mma_fp8(a[mt],b[nt],acc[mt][nt]);
    }
    __syncthreads();
  }
  const int r0=lane>>2, cg=(lane&3)*2;
  #pragma unroll
  for(int mt=0;mt<2;mt++)
    #pragma unroll
    for(int nt=0;nt<4;nt++){
      int g0=m_blk+wm*32+mt*16+r0, g1=g0+8, gn=n_blk+wn*32+nt*8+cg;
      if(g0<M&&gn<N){D[(size_t)g0*N+gn]=acc[mt][nt][0]; if(gn+1<N)D[(size_t)g0*N+gn+1]=acc[mt][nt][1];}
      if(g1<M&&gn<N){D[(size_t)g1*N+gn]=acc[mt][nt][2]; if(gn+1<N)D[(size_t)g1*N+gn+1]=acc[mt][nt][3];}
    }
}

int main(int argc,char**argv){
  int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]),it=argc>4?atoi(argv[4]):50;
  std::vector<uint8_t>A((size_t)M*K),B((size_t)N*K);
  for(size_t i=0;i<A.size();i++)A[i]=(uint8_t)(0x30+((i*7)%8));
  for(size_t i=0;i<B.size();i++)B[i]=(uint8_t)(0x30+((i*5)%8));
  uint8_t *dA,*dB;float* dD;
  cudaMalloc(&dA,A.size());cudaMalloc(&dB,B.size());cudaMalloc(&dD,(size_t)M*N*4);
  cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice);
  dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
  gemm_v2<<<grid,256>>>(dA,dB,dD,M,N,K);
  auto e=cudaDeviceSynchronize();
  if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}
  if(M<=128&&N<=128&&K<=256){
    std::vector<float>D((size_t)M*N);cudaMemcpy(D.data(),dD,(size_t)M*N*4,cudaMemcpyDeviceToHost);
    int bad=0;double mx=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
      double s=0;for(int kk=0;kk<K;kk++)s+=(double)e4m3_f(A[(size_t)m*K+kk])*e4m3_f(B[(size_t)n*K+kk]);
      double d=fabs(s-D[(size_t)m*N+n]); if(d>1e-2*fmax(1.0,fabs(s)))bad++; mx=fmax(mx,d);
    }
    printf("PARITY bad=%d/%d maxabs=%.5f (D[0]=%.3f)\n",bad,M*N,mx,D[0]);
  }
  cudaEvent_t s0,s1;cudaEventCreate(&s0);cudaEventCreate(&s1);
  cudaEventRecord(s0); for(int i=0;i<it;i++) gemm_v2<<<grid,256>>>(dA,dB,dD,M,N,K);
  cudaEventRecord(s1);cudaEventSynchronize(s1);
  float ms;cudaEventElapsedTime(&ms,s0,s1);ms/=it;
  printf("M=%d N=%d K=%d | %.3f ms | %.1f TF/s\n",M,N,K,ms,2.0*M*N*K/(ms*1e-3)/1e12);
  return 0;
}
