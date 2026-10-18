// CUDA IPC 探针：server 在 dev A 分配并导出 handle 到文件；client 在 dev B 打开并拷贝。
// 用法：probe_ipc A B   （内部用 system() 起 client，避免 fork 后用 CUDA）
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N (64u << 20)
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("  %s -> %s\n", #x, cudaGetErrorString(e)); return 2; } } while (0)
static int client(int b, const char *f) {
    cudaIpcMemHandle_t h; FILE *fp = fopen(f, "rb"); if (!fp || fread(&h, sizeof h, 1, fp) != 1) return 9; fclose(fp);
    CK(cudaSetDevice(b));
    void *q; cudaError_t e = cudaIpcOpenMemHandle(&q, h, cudaIpcMemLazyEnablePeerAccess);
    printf("  open on dev %d: %s\n", b, cudaGetErrorString(e)); if (e) return 3;
    void *d; CK(cudaMalloc(&d, N)); cudaEvent_t s, t; cudaEventCreate(&s); cudaEventCreate(&t);
    cudaMemcpy(d, q, N, cudaMemcpyDefault); // warm
    cudaEventRecord(s); e = cudaMemcpy(d, q, N, cudaMemcpyDefault); cudaEventRecord(t); cudaEventSynchronize(t);
    float ms; cudaEventElapsedTime(&ms, s, t);
    unsigned char v = 0; cudaMemcpy(&v, d, 1, cudaMemcpyDeviceToHost);
    printf("  memcpy ipc->local 64MB: %s, %.3f ms, %.1f GB/s, byte=%02x\n", cudaGetErrorString(e), ms, N / ms / 1e6, v);
    cudaIpcCloseMemHandle(q); return e ? 4 : 0;
}
int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "client")) return client(atoi(argv[2]), argv[3]);
    int a = atoi(argv[1]), b = atoi(argv[2]); int can = -1;
    cudaDeviceCanAccessPeer(&can, b, a); printf("IPC %d -> %d: canAccessPeer(%d,%d)=%d\n", a, b, b, a, can);
    CK(cudaSetDevice(a)); void *p; CK(cudaMalloc(&p, N)); CK(cudaMemset(p, 0x5a, N)); CK(cudaDeviceSynchronize());
    cudaIpcMemHandle_t h; CK(cudaIpcGetMemHandle(&h, p));
    char f[64]; snprintf(f, sizeof f, "/tmp/ipc_%d_%d.h", a, b); FILE *fp = fopen(f, "wb"); fwrite(&h, sizeof h, 1, fp); fclose(fp);
    char cmd[256]; snprintf(cmd, sizeof cmd, "%s client %d %s", argv[0], b, f);
    int st = system(cmd); printf("  client exit %d\n", WEXITSTATUS(st)); return 0;
}
