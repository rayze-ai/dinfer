#!/usr/bin/env python3
"""把两份 nsys kernel 汇总（`nsys stats --report cuda_gpu_kern_sum --format csv`）按阶段归类并并排输出 Markdown。
用法：nsys_stage_compare.py dinfer_kern.csv vllm_kern.csv [--steps N_dinfer N_vllm]
阶段归类靠 kernel 名关键字（见 STAGES）；归不进的进 "other"。给 --steps 时额外输出每 step 的毫秒数。
"""
import csv, re, sys

STAGES = [
    ("attention", r"gqa_|flash|attn|paged_attention|reshape_and_cache|_attention|fmha|fused_attention|MHA|triton_.*attn"),
    ("gemm", r"gemm|cutlass|cublas|Kernel2<|ampere_|sm80_|sm90_|xmma|gemv"),
    ("norm", r"rmsnorm|rms_norm|layernorm|layer_norm|fused_add"),
    ("act/mlp", r"silu|gelu|act_and_mul|mul_kernel|elementwise_kernel.*mul"),
    ("rope/embed", r"rope|rotary|embedding|gather_rows|index_select|indexing"),
    ("sampling", r"sample|argmax|topk|top_k|top_p|softmax|logprob|penalt|temperature|min_p|multinomial|sort|radix|cub::"),
    ("copy/misc", r"copy_cols|copy|memcpy|memset|fill|cat_|concat|transpose|permute|elementwise|vectorized"),
]

def load(path):
    rows = list(csv.reader(open(path)))
    hdr = next((i for i, r in enumerate(rows) if r and r[0].startswith("Time")), None)
    out = []
    if hdr is None:
        return out
    for r in rows[hdr + 1:]:
        try:
            out.append((float(r[1]) / 1e6, int(r[2]), r[-1]))  # total ms, count, name
        except Exception:
            pass
    return out

def classify(name):
    for stage, pat in STAGES:
        if re.search(pat, name, re.I):
            return stage
    return "other"

def summarize(rows):
    agg = {}
    for ms, n, name in rows:
        st = classify(name)
        a = agg.setdefault(st, [0.0, 0, {}])
        a[0] += ms; a[1] += n
        a[2][name[:70]] = a[2].get(name[:70], 0.0) + ms
    return agg

def main():
    args = sys.argv[1:]
    steps = None
    if "--steps" in args:
        i = args.index("--steps"); steps = (int(args[i + 1]), int(args[i + 2])); del args[i:i + 3]
    a, b = summarize(load(args[0])), summarize(load(args[1]))
    ta = sum(v[0] for v in a.values()); tb = sum(v[0] for v in b.values())
    print("| 阶段 | dinfer ms (%) | vLLM ms (%) | dinfer kernel 次数 | vLLM 次数 |" + (" dinfer ms/step | vLLM ms/step |" if steps else ""))
    print("|---|---|---|---|---|" + ("---|---|" if steps else ""))
    for st, _ in STAGES + [("other", "")]:
        x = a.get(st, [0, 0, {}]); y = b.get(st, [0, 0, {}])
        row = f"| {st} | {x[0]:.1f} ({100*x[0]/max(ta,1e-9):.0f}%) | {y[0]:.1f} ({100*y[0]/max(tb,1e-9):.0f}%) | {x[1]} | {y[1]} |"
        if steps:
            row += f" {x[0]/steps[0]:.3f} | {y[0]/steps[1]:.3f} |"
        print(row)
    print(f"| **total GPU** | {ta:.1f} | {tb:.1f} | {sum(v[1] for v in a.values())} | {sum(v[1] for v in b.values())} |" + (f" {ta/steps[0]:.3f} | {tb/steps[1]:.3f} |" if steps else ""))
    for label, agg in (("dinfer", a), ("vLLM", b)):
        print(f"\n<details><summary>{label} top kernels</summary>\n")
        items = sorted(((ms, st, n) for st, v in agg.items() for n, ms in v[2].items()), reverse=True)[:12]
        for ms, st, n in items:
            print(f"- {ms:8.1f} ms  [{st}]  `{n}`")
        print("</details>")

if __name__ == "__main__":
    main()
