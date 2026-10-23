# kernel 变体下限实测（2026-10-23）

结论先行：**逐位复刻 vLLM 不可达**；但**我们从 L2 起比 vLLM 自家 kernel 变体差 2–3 倍，这是尚待定位的具体问题**。

## 有效性验证（这次逐条确认过）

- 强制切换生效：`md5sum fk4_cutlass/L29.bin` = `1a11ab15…` ≠ `fk4_flashinfer/L29.bin` = `987335c6…`
- 两轮日志都有 `写 30 个 .bin`
- 跑前跑后 `nvidia-smi` 8 卡全空，0 容器

命令（容器内，GPU 4）：

```bash
FORCE_KERNEL=flashinfer OUT=/w/hid/fk4_flashinfer \
  python3 -I scripts/dev/kaon3_vllm_hidden.py <model> $(cat test/kaon3/oracle_clean/p02.ids)
FORCE_KERNEL=cutlass    OUT=/w/hid/fk4_cutlass    \
  python3 -I scripts/dev/kaon3_vllm_hidden.py <model> $(cat test/kaon3/oracle_clean/p02.ids)
```

两轮都命中 120 个 fp8 linear 层（脚本打印 `picked kernels: {'…Kernel': 120}`）。

## 逐层相对误差（p02，24 token）

| 层 | vLLM 自家 fi-vs-cu | 我们 vs vLLM（D-2）|
|---|---|---|
| L-1 | 0 | 0 |
| L0 | 2.649e-05 | 4.9e-08 |
| L1 | 4.278e-03 | 1.3e-06 |
| L2 | 8.525e-03 | 2.07e-02 |
| L3 | 1.292e-02 | 3.87e-02 |
| L5 | 1.644e-02 | 5.35e-02 |
| L8 | 2.544e-02 | 8.35e-02 |
| L11 | 3.548e-02 | 1.07e-01 |
| L14 | 4.781e-02 | 1.30e-01 |
| L17 | 5.497e-02 | 1.29e-01 |
| L20 | 5.728e-02 | 1.30e-01 |
| L23 | 6.570e-02 | 1.36e-01 |
| L26 | 1.053e-01 | 1.60e-01 |
| L29 | 9.382e-02 | 2.93e-01 |

## 解读

1. **L0：我们 4.9e-8，比 vLLM 自家两 kernel 的 2.65e-5 好 540 倍** → 我们的 fp8 GEMM 与它默认的 flashinfer 路径高度一致。
2. **L2 起反超 2–3 倍** → 我们这边还有一个放大器是 vLLM 没有的，或同一放大器系数更陡。
3. 单层三方对比（L0 `q_proj`，同输入同权重）：我们 vs 参照前向 5/98304；参照自己调自己的 flashinfer 算子 7/196608；
   fi-vs-cu 10/196608 → **参照自己也做不到单算子复现自己的前向**。

## 原始输出（节选）

```
层      fi-vs-cu rel   max          diff
L0      2.649e-05      1.875e-01    764/67584
L1      4.278e-03      1.641e+00    34476/67584
L2      8.525e-03      1.188e+00    58646/67584
L11     3.548e-02      4.750e+00    65179/67584
L29     9.382e-02      1.975e+01    67057/67584
```

（md5 校验：`1a11ab15b7c691818b37be9df73b0ed2` cutlass / `987335c696ebc996266d97215d7c4a4e` flashinfer）
