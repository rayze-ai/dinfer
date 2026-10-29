#!/usr/bin/env bash
# b61（D231 复测）：同二进制 ROUTE_SPLIT_MIN_TOK 9（A）/ 0（B），顺序反过来（先 split 后 fused）看 b59 的 c16 −3% 是否顺序效应。
TAG=b61ab AENV="ROUTE_SPLIT_MIN_TOK=9" BENV="ROUTE_SPLIT_MIN_TOK=0" PORT=8358 bash /w/envab_c.sh
