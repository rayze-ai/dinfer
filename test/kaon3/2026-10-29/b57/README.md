# b57：五刀 + B50 二进制上 c16 纯 decode 步 nsys（`--cuda-graph-trace=node`，graph id=14 = M=16 桶，189 步）
`RESULT.txt` 为 `nsys_kern.py --all` 输出；逐段表与判定见 loop D227。要点：MoE 2.8 ms/步已在带宽下限；小 M fp8 GEMM（gemv<16> 1.9 + gateup_act 0.73 + nvjet 0.41）3.0 ms 离下限 0.8 ms 是靶子。
