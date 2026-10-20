# K3 C-5：多卡切分 TP + EP（tp2 × 前 5 层对拍 + serve 冒烟）

## 目标
`models/kimik3` 接 `(rank, world)` 分片装载 + 通信点，world=1 逐位不变；B300 GPU 2–3 tp2 × 前 5 层：`model_parity` 与单卡同一套 HF refs 对拍，再 `--tp 2` serve 冒烟。切分表 `docs/design-kimik3.md` §2.4。

## 环境
- B300-50（`B300_TAG=50 scripts/dev/remote_b300.sh run ...`）；`source /data/yaochen/infer/env.sh`；NCCL 2.30.3 在 `/usr/lib/x86_64-linux-gnu/libnccl.so.2`（系统自带，不用拷）。
- GPU：`CUDA_DEVICE_ORDER=PCI_BUS_ID`，**只用 2–3**（0/1 Qwen3.5 常驻，4–7 另一子代理）。
- 权重 `/data/models/moonshotai/Kimi-K3`；refs `test/kimik3/2026-10-20/model/`（已在远端）。
- 生成物放 `$R/bootstrap/k3tp/`。

## 观察方法
1. 单卡基线：`K3_SKIP_GEN=1 scripts/verify/k3_model.sh`（GPU 2）→ 隔离 cos / 链式 cos / logits cos / top-1。
2. tp2：`K3_SKIP_GEN=1 K3_TP=2 K3_GPU=2,3 scripts/verify/k3_model.sh`（测试内 `KIMIK3_TP=2` → 每 rank 一线程，`Device::new(rank)` + `Comm::init`；只 rank 0 比对）。
3. 两份结果对比：隔离口径每层 cos 与单卡差 ≤ 1e-4（即对 HF 都 ≥ 0.999），logits cos ≥ 0.9998，top-1 一致；`KIMIK3_DUMP=dir` 把本机各层输出写 .npy，tp2 vs 单卡直接 cos ≥ 0.9999。
4. serve：`dinfer serve --model ... --layers 5 --tp 2 --device 2 --port 8014`，流式一条 + c1 bench 8 条（`scripts/verify/k3_smoke.sh` 若有，否则 curl）。

## 停止条件
- `k3_model.sh`（world=1）exit 0 不退化；`K3_TP=2` exit 0；tp2 vs 单卡逐层 cos ≥ 0.9999，logits cos ≥ 0.9998，top-1 同。
- `--tp 2` serve 流式一条正常收尾 + 8 条 c1 bench 不报错。
- 四 feature 构建过；`cargo test -p dinfer-kimik3` 过。
- 迭代上限：对拍 5 轮。

## 记录
- 第 1 轮（代码，本机）：commit 77f8b91（§2.4 表 + 本 loop）、e02ffc7（分片 + 通信点 + tp executor + service/bin）、2ca2d29（model_parity KIMIK3_TP / KIMIK3_DUMP + k3_model.sh K3_TP + k3_tp_compare.py）。kimik3 bin 构建过、`cargo test -p dinfer-kimik3` 24 过、ref-tests `--no-run` 过。
  - 设计取舍：EP v1 不 all-gather（TP 下隐状态已全量）；MoE 两次 all-reduce（acc f32 在 norm 前——norm 非线性不能后移；shared 部分和 bf16）；embed / lm_head 复制。v2 可把两次合成一次 f32 all-reduce（`[acc | shared]` 拼接）。
  - 阻塞：B300 token 过期（`run` 无输出、`put` rsync code 12），已向主会话要新 token；待传文件列表见 progress。
- 第 2 轮准备（10:41）：主会话拿到的 token 到手即过期（10:44 +0800 到期，拿到时 10:41），远端 push 失败；我误 kill 了陈旧 master（之后不再碰 ControlMaster，只用 run / put）。B300 待跑清单（约 20 min GPU）：① 远端 `cargo build --release -p dinfer -p bench --features kimik3` + `model_parity --no-run`；② 单卡 `K3_SKIP_GEN=1 K3_GPU=2 K3_DUMP=$R/bootstrap/k3tp/single scripts/verify/k3_model.sh`；③ `K3_TP=2 K3_GPU=2,3 K3_DUMP=$R/bootstrap/k3tp/tp2 ...`；④ `k3_tp_compare.py single tp2`；⑤ `K3_GPU=2,3 K3_TP=2 bash scripts/dev/k3_b300_smoke.sh`。`k3_b300_smoke.sh` 已支持 `K3_TP`（e67cb8f）。
