// FP8 GEMM 基准：smem 分块，8 warp = 2(M) x 4(N)，每 warp 16x16（2 个 n8）
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
__device__ __forceinline__ void mma_fp8(const uint32_t* a,const uint32_t* b,float* d){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
constexpr int BM=32, BN=64, BK=128;
__global__ void gemm(const uint8_t* __restrict__ A,const uint8_t* __restrict__ B,
                     float* __restrict__ D,int M,int N,int K){
  __shared__ uint8_t sA[BM*BK], sB[BN*BK];
  const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  const int wm=warp/4, wn=warp%4;
  const int m_blk=blockIdx.y*BM, n_blk=blockIdx.x*BN;
  float acc[2][4]={{0,0,0,0},{0,0,0,0}};
  const int nk=K/BK;
  for(int kb=0;kb<nk;kb++){
    { const uint4* src=reinterpret_cast<const uint4*>(A+(size_t)m_blk*K+kb*BK);
      uint4* dst=reinterpret_cast<uint4*>(sA);
      const int per=BM*BK/16, rb=BK/16;
      for(int i=tid;i<per;i+=blockDim.x) dst[i]=src[(size_t)(i/rb)*(K/16)+(i%rb)]; }
    { const uint4* src=reinterpret_cast<const uint4*>(B+(size_t)n_blk*K+kb*BK);
      uint4* dst=reinterpret_cast<uint4*>(sB);
      const int per=BN*BK/16, rb=BK/16;
      for(int i=tid;i<per;i+=blockDim.x) dst[i]=src[(size_t)(i/rb)*(K/16)+(i%rb)]; }
    __syncthreads();
    for(int b=0;b<BK/32;b++){
      const int kb0=b*32, mr=wm*16, nc=wn*16;
      uint32_t a[4];
      int r=lane/4, cq=(lane%4)*4;
      a[0]=*reinterpret_cast<const uint32_t*>(sA+(mr+r)*BK+kb0+cq);
      a[1]=*reinterpret_cast<const uint32_t*>(sA+(mr+r+8)*BK+kb0+cq);
      a[2]=*reinterpret_cast<const uint32_t*>(sA+(mr+r)*BK+kb0+cq+16);
      a[3]=*reinterpret_cast<const uint32_t*>(sA+(mr+r+8)*BK+kb0+cq+16);
      #pragma unroll
      for(int q=0;q<2;q++){
        uint32_t bb[2];
        bb[0]=*reinterpret_cast<const uint32_t*>(sB+(nc+q*8+r)*BK+kb0+cq);
        bb[1]=*reinterpret_cast<const uint32_t*>(sB+(nc+q*8+r)*BK+kb0+cq+16);
        mma_fp8(a,bb,acc[q]);
      }
    }
    __syncthreads();
  }
  int r0=lane/4,cg=(lane%4)*2;
  #pragma unroll
  for(int q=0;q<2;q++){
    int g0=m_blk+wm*16+r0,g1=g0+8,gn=n_blk+wn*16+q*8+cg;
    if(g0<M&&gn<N){D[(size_t)g0*N+gn]=acc[q][0]; if(gn+1<N)D[(size_t)g0*N+gn+1]=acc[q][1];}
    if(g1<M&&gn<N){D[(size_t)g1*N+gn]=acc[q][2]; if(gn+1<N)D[(size_t)g1*N+gn+1]=acc[q][3];}
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
  gemm<<<grid,256>>>(dA,dB,dD,M,N,K);
  auto e=cudaDeviceSynchronize(); if(e){printf("err %s\n",cudaGetErrorString(e));return 1;}
  cudaEvent_t s0,s1;cudaEventCreate(&s0);cudaEventCreate(&s1);
  cudaEventRecord(s0); for(int i=0;i<it;i++) gemm<<<grid,256>>>(dA,dB,dD,M,N,K);
  cudaEventRecord(s1);cudaEventSynchronize(s1);
  float ms;cudaEventElapsedTime(&ms,s0,s1);ms/=it;
  printf("M=%d N=%d K=%d | %.3f ms | %.1f TF/s\n",M,N,K,ms,2.0*M*N*K/(ms*1e-3)/1e12);
  return 0;
}
