// 用 CUTLASS 的 MMA_Traits 自动生成寄存器布局，验证 SM120 NVFP4 MMA 的数值正确性。
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>

using namespace cute;

template<class T> __host__ T h_from_bits(uint8_t b);
template<> __host__ float h_from_bits<float>(uint8_t b){ return 0.f; }

__host__ __device__ inline float e2m1_f(unsigned n){
  static const float t[8]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  float m=t[n&7u]; return (n&8u)? -m:m;
}
__host__ __device__ inline float e4m3_f(unsigned b){
  unsigned s=(b>>7)&1u,e=(b>>3)&15u,m=b&7u;
  float v = (e==0)? (m*(1.f/8.f)*powf(2.f,-6.f)) : (1.f+m*(1.f/8.f))*powf(2.f,(int)e-7);
  return s?-v:v;
}

using Op = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<uint4_t,uint4_t,float,uint8_t,16>;
using Tr = MMA_Traits<Op>;
using TA = typename Tr::ALayout;
using TB = typename Tr::BLayout;
using TSF = typename Tr::SFALayout;

__global__ void k(const uint8_t* A, const uint8_t* B, const uint8_t* sfa, const uint8_t* sfb, float* D){
  // A/B 在 smem 里按逻辑 [M,K]/[N,K] 打包（每字节两个 fp4，低半字节在前）
  extern __shared__ uint8_t sm[];
  uint8_t* sA=sm; uint8_t* sB=sm+16*64/2; uint8_t* sSFA=sB+8*64/2;
  uint8_t* sSFB=sSFA+16;
  for(int i=threadIdx.x;i<16*64/2;i+=blockDim.x) sA[i]=A[i];
  for(int i=threadIdx.x;i<8*64/2;i+=blockDim.x) sB[i]=B[i];
  for(int i=threadIdx.x;i<16;i+=blockDim.x) sSFA[i]=sfa[i];
  for(int i=threadIdx.x;i<8;i+=blockDim.x) sSFB[i]=sfb[i];
  __syncthreads();
  auto gA=make_tensor(make_smem_ptr(reinterpret_cast<uint4_t*>(sA)), make_layout(make_shape(Int<16>{},Int<64>{}),LayoutLeft{}));
  auto gB=make_tensor(make_smem_ptr(reinterpret_cast<uint4_t*>(sB)), make_layout(make_shape(Int<8>{},Int<64>{}),LayoutLeft{}));
  auto gSA=make_tensor(make_smem_ptr(sSFA), make_layout(make_shape(Int<16>{},Int<4>{}),LayoutLeft{}));
  auto gSB=make_tensor(make_smem_ptr(sSFB), make_layout(make_shape(Int<8>{},Int<4>{}),LayoutLeft{}));
  uint32_t a[4]={0,0,0,0}, b[2]={0,0};
  // 按 traits 的 thread/value -> (M,K) 映射取值，自己按位拼
  // 实测映射（show_layout 打印）：A linear = 128*(t%4) + (t/4) + 16*v0 + 8*v1 + 512*v2，
  // 在 (M16,K64) 列主序下的 (m,k) = (linear%16, linear/16)
  for(int v=0;v<32;v++){ int v0=v%8, v1=(v/8)%2, v2=v/16;
    int lin = 128*(threadIdx.x%4) + (threadIdx.x/4) + 16*v0 + 8*v1 + 512*v2;
    int m = lin/64, kk = lin%64;   // 行主序 (M16,K64)
    uint8_t byte=sA[(m*64+kk)/2]; unsigned nib=(kk%2)?((byte>>4)&15):(byte&15);
    a[v/8] |= (uint32_t)nib << (4*(v%8)); }
  for(int v=0;v<16;v++){ int v0=v%8, v1=v/8;
    int lin = 64*(threadIdx.x%4) + (threadIdx.x/4) + 8*v0 + 256*v1;
    int n = lin/64, kk = lin%64;   // 行主序 (N8,K64)
    uint8_t byte=sB[(n*64+kk)/2]; unsigned nib=(kk%2)?((byte>>4)&15):(byte&15);
    b[v/8] |= (uint32_t)nib << (4*(v%8)); }
  // SFA linear = 8*(t%2) + (t/4) + 16*v ；scale 张量形状 [M, K/16]，列主序 idx = m*4 + v
  uint32_t sa=0, sb=0;
  // 4X：一个 u32 装 4 个 k-block 的 e4m3 字节（k=0..15,16..31,32..47,48..63）
  { int lin = 8*(threadIdx.x%2) + (threadIdx.x/4);
    int m = lin%16;
    sa = (uint32_t)sSFA[m*4+0] | ((uint32_t)sSFA[m*4+1]<<8) | ((uint32_t)sSFA[m*4+2]<<16) | ((uint32_t)sSFA[m*4+3]<<24); }
  { int lin = (threadIdx.x/4);           // SFBLayout stride (_0,_1): n 由 t/4 决定
    int n = lin%8;
    sb = sSFB[n*4 + 0]; }
  float d0=0,d1=0,d2=0,d3=0,c0=0,c1=0,c2=0,c3=0;
  uint16_t bid=0,tid=0;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(sa),"h"(bid),"h"(tid),"r"(sb),"h"(bid),"h"(tid));
  // CLayout SM80_16x8_Row
  int r0=threadIdx.x/4, cg=(threadIdx.x%4)*2;
  D[r0*8+cg]=d0; D[r0*8+cg+1]=d1; D[(r0+8)*8+cg]=d2; D[(r0+8)*8+cg+1]=d3;
}

int main(){
  const int M=16,N=8,K=64;
  std::vector<uint8_t> A(M*K/2),B(N*K/2),SFA(M),SFB(N);
  std::vector<float> Af(M*K),Bf(N*K);
  // 探针：A 全 1.0(码 2)，B 全 1.0(码 2) → D 应全 64
  for(int i=0;i<M*K;i++){ int n=2; Af[i]=e2m1_f(n); if(i%2==0)A[i/2]=n; else A[i/2]|=(n<<4);}
  for(int i=0;i<N*K;i++){ int n=2; Bf[i]=e2m1_f(n); if(i%2==0)B[i/2]=n; else B[i/2]|=(n<<4);}
  for(int m=0;m<M;m++) SFA[m]=0x38;
  for(int n=0;n<N;n++) SFB[n]=0x38;
  uint8_t *dA,*dB,*dSFA,*dSFB; float* dD;
  cudaMalloc(&dA,A.size());cudaMalloc(&dB,B.size());cudaMalloc(&dSFA,SFA.size());
  cudaMalloc(&dSFB,SFB.size());cudaMalloc(&dD,M*N*4);
  cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFA,SFA.data(),SFA.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFB,SFB.data(),SFB.size(),cudaMemcpyHostToDevice);
  cudaMemset(dD,0,M*N*4);
  k<<<1,32,4096>>>(dA,dB,dSFA,dSFB,dD);
  auto e=cudaDeviceSynchronize();
  printf("kernel: %s\n",cudaGetErrorString(e));
  std::vector<float> D(M*N); cudaMemcpy(D.data(),dD,M*N*4,cudaMemcpyDeviceToHost);
  double mx=0; int bad=0;
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){
    double s=0; for(int kk=0;kk<K;kk++) s+=(double)Af[m*K+kk]*Bf[n*K+kk];
    s*=e4m3_f(SFA[m])*e4m3_f(SFB[n]);
    double d=fabs(s-D[m*N+n]); if(d>1e-3*fmax(1.0,fabs(s))) bad++; mx=fmax(mx,d);
    if((m<2||m==8)&&n<2) printf("D[%d,%d]=%.5f ref=%.5f\n",m,n,D[m*N+n],s);
  }
  printf("maxabs=%.5f bad=%d/%d\n",mx,bad,M*N);
  return 0;
}
