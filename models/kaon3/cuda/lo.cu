#include <cstdio>
#include <cuda_runtime.h>
__global__ void empty(){}
__global__ void readb(const uint4* p,uint4* o,long n){long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=p[i];}
int main(){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  empty<<<1,32>>>(); cudaDeviceSynchronize();
  cudaEventRecord(a); for(int i=0;i<1000;i++) empty<<<1,32>>>(); cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b); printf("empty kernel: %.3f us\n", ms);
  for(long mb : {2L, 32L, 256L, 1024L}){
    long n=mb*1024*1024/16; uint4 *p,*o; cudaMalloc(&p,mb*1024*1024); cudaMalloc(&o,mb*1024*1024);
    readb<<<(n+255)/256,256>>>(p,o,n); cudaDeviceSynchronize();
    cudaEventRecord(a); for(int i=0;i<20;i++) readb<<<(n+255)/256,256>>>(p,o,n); cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms,a,b); ms/=20;
    printf("read %4ld MB: %7.3f us -> %.0f GB/s\n", mb, ms*1000, mb*2/(ms*1e-3)/1e3);
    cudaFree(p); cudaFree(o);
  }
  return 0;
}
