#!/usr/bin/env bash
# B3 逐位对拍：host-loop MoE（MOE_GROUPED=0）vs device-grouped（=1），比 moe_y / ff2。
# 在 ctr 容器内单次调用跑完（/tmp 不跨 ctr 持久）。
set -u
D=/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef12555d299ed7ca02ade5e77f239f27478
cd /w/dinfer
GPU=${GPU:-0}
IDS="${IDS:-1 2 3 4 5 6 7 8}"
B=/w/dinfer/.b3_base
G=/w/dinfer/.b3_grp
rm -rf "$B" "$G"; mkdir -p "$B" "$G"
echo "=== baseline (MOE_GROUPED=0) ==="
CUDA_VISIBLE_DEVICES=$GPU SUB_LAYER=0 MOE_GROUPED=0 ./target/release/kaon3_probe "$D" --forward --hidden-all "$B" $IDS 2>&1 | tail -3
echo "=== grouped (MOE_GROUPED=1) ==="
CUDA_VISIBLE_DEVICES=$GPU SUB_LAYER=0 MOE_GROUPED=1 ./target/release/kaon3_probe "$D" --forward --hidden-all "$G" $IDS 2>&1 | tail -3
echo "=== diff ==="
for f in moe_y ff2 pre_ff2_norm router_logits; do
  if [ -f "$B/$f.bin" ] && [ -f "$G/$f.bin" ]; then
    if cmp -s "$B/$f.bin" "$G/$f.bin"; then echo "$f: IDENTICAL 0/N"; else
      n=$(cmp -l "$B/$f.bin" "$G/$f.bin" 2>/dev/null | wc -l)
      tot=$(stat -c%s "$B/$f.bin"); echo "$f: DIFF ndiff_bytes=$n / $tot bytes";
    fi
  else echo "$f: MISSING (B=$([ -f $B/$f.bin ] && echo y||echo n) G=$([ -f $G/$f.bin ] && echo y||echo n))"; fi
done
echo "=== files in base ==="; ls "$B" | tr '\n' ' '; echo
