// 用 CUTLASS/CuTe 的 make_fragment + copy 填寄存器（内部即正确布局），
// 验证 SM120 NVFP4 block-scaled MMA 的数值。
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>

using namespace cute;

__host__ __device__ inline float e2m1_f(unsigned n){
  static const float t[8]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
  float m=t[n&7u]; return (n&8u)? -m:m;
}
__host__ __device__ inline float e4m3_f(unsigned b){
  unsigned s=(b>>7)&1u,e=(b>>3)&15u,m=b&7u;
  float v=(e==0)?(m*(1.f/8.f)*powf(2.f,-6.f)):(1.f+m*(1.f/8.f))*powf(2.f,(int)e-7);
  return s?-v:v;
}

using Op = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<float_e2m1_t,float_e2m1_t,float,float_ue4m3_t,16>;
using Tr = MMA_Traits<Op>;

__global__ void k(const uint8_t* gA,const uint8_t* gB,const uint8_t* gSA,const uint8_t* gSB,float* gD){
  extern __shared__ uint8_t sm[];
  float_e2m1_t* sA = reinterpret_cast<float_e2m1_t*>(sm);
  float_e2m1_t* sB = sA + 16*64;
  uint8_t* sSA = sm + (16*64+8*64)/2;
  uint8_t* sSB = sSA + 16*4;
  // 逻辑 [M,K] 行主序打包：每字节两值，低半字节在前
  {
    uint8_t* pA=reinterpret_cast<uint8_t*>(sA); uint8_t* pB=reinterpret_cast<uint8_t*>(sB);
    for(int i=threadIdx.x;i<16*64/2;i+=blockDim.x) pA[i]=gA[i];
    for(int i=threadIdx.x;i<8*64/2;i+=blockDim.x) pB[i]=gB[i];
  }
  for(int i=threadIdx.x;i<64;i+=blockDim.x) sSA[i]=gSA[i];
  for(int i=threadIdx.x;i<32;i+=blockDim.x) sSB[i]=gSB[i];
  __syncthreads();

  auto LA = make_layout(make_shape(Int<16>{},Int<64>{}), GenRowMajor{});
  auto LB = make_layout(make_shape(Int<8>{},Int<64>{}), GenRowMajor{});
  auto LSA= make_layout(make_shape(Int<16>{},Int<4>{}), GenRowMajor{});
  auto LSB= make_layout(make_shape(Int<8>{},Int<4>{}), GenRowMajor{});
  Tensor tA = make_tensor(make_smem_ptr(sA), LA);
  Tensor tB = make_tensor(make_smem_ptr(sB), LB);
  Tensor tSA= make_tensor(make_smem_ptr(sSA), LSA);
  Tensor tSB= make_tensor(make_smem_ptr(sSB), LSB);

  TiledMMA tiled = make_tiled_mma(Op{});
  auto thr = tiled.get_slice(threadIdx.x);

  auto fragA = thr.partition_A(tA);         // (MMA, MMA_M, MMA_K)
  auto fragB = thr.partition_B(tB);
  auto fragSA= thr.partition_A(tSA);
  auto fragSB= thr.partition_B(tSB);

  Tensor rA = make_fragment_like(fragA);
  Tensor rB = make_fragment_like(fragB);
  Tensor rSA= make_fragment_like(fragSA);
  Tensor rSB= make_fragment_like(fragSB);
  copy(fragA, rA); copy(fragB, rB);
  copy(fragSA, rSA); copy(fragSB, rSB);

  Tensor rC = make_tensor<float>(make_shape(Int<16>{},Int<8>{}));
  clear(rC);
  auto fragC = thr.partition_C(rC);
  Tensor rD = make_fragment_like(fragC);
  clear(rD);
  // 直接调 atom 的 fma，绕开 gemm() 重载
  {
    uint32_t* pa = reinterpret_cast<uint32_t*>(rA.data().get());
    uint32_t* pb = reinterpret_cast<uint32_t*>(rB.data().get());
    float* pc = reinterpret_cast<float*>(rD.data().get());
    uint32_t* psa = reinterpret_cast<uint32_t*>(rSA.data().get());
    uint32_t* psb = reinterpret_cast<uint32_t*>(rSB.data().get());
    Op::fma(pc[0],pc[1],pc[2],pc[3],
            pa[0],pa[1],pa[2],pa[3],
            pb[0],pb[1],
            0.f,0.f,0.f,0.f,
            psa[0], psb[0]);
  }

  // 写回：用 CLayout 的 thread/value 映射
  auto CL = typename Tr::CLayout{};
  int nr = size(rD);
  for(int i=0;i<nr;i++){
    int lin = CL(threadIdx.x, i);   // -> (M16,N8) 行主序
    gD[lin] = rD(i);
  }
}

int main(){
  const int M=16,N=8,K=64;
  std::vector<uint8_t> A(M*K/2),B(N*K/2),SA(M*4),SB(N*4);
  std::vector<float> Af(M*K),Bf(N*K);
  for(int i=0;i<M*K;i++){ int n=(i*7+3)&15; Af[i]=e2m1_f(n); if(i%2==0)A[i/2]=n; else A[i/2]|=(n<<4);}
  for(int i=0;i<N*K;i++){ int n=(i*5+1)&15; Bf[i]=e2m1_f(n); if(i%2==0)B[i/2]=n; else B[i/2]|=(n<<4);}
  for(int i=0;i<M*4;i++) SA[i]=0x38;
  for(int i=0;i<N*4;i++) SB[i]=0x38;
  uint8_t *dA,*dB,*dSA,*dSB; float* dD;
  cudaMalloc(&dA,A.size());cudaMalloc(&dB,B.size());cudaMalloc(&dSA,SA.size());
  cudaMalloc(&dSB,SB.size());cudaMalloc(&dD,M*N*4);
  cudaMemcpy(dA,A.data(),A.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,B.data(),B.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSA,SA.data(),SA.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSB,SB.data(),SB.size(),cudaMemcpyHostToDevice);
  cudaMemset(dD,0,M*N*4);
  k<<<1,32,16384>>>(dA,dB,dSA,dSB,dD);
  printf("kernel: %s\n",cudaGetErrorString(cudaDeviceSynchronize()));
  std::vector<float> D(M*N); cudaMemcpy(D.data(),dD,M*N*4,cudaMemcpyDeviceToHost);
  double mx=0; int bad=0;
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){
    double s=0; for(int kk=0;kk<K;kk++) s+=(double)Af[m*K+kk]*Bf[n*K+kk];
    double d=fabs(s-D[m*N+n]); if(d>1e-2*fmax(1.0,fabs(s))) bad++; mx=fmax(mx,d);
    if(m==0&&n<2) printf("D[%d,%d]=%.4f ref=%.4f\n",m,n,D[m*N+n],s);
  }
  printf("maxabs=%.4f bad=%d/%d\n",mx,bad,M*N);
  return 0;
}
