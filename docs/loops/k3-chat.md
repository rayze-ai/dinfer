# Kimi-K3 里程碑 C-4a：tokenizer（tiktoken）+ XTML 聊天渲染 + 输出流式解析

> 状态：**完成（2026-10-20，`scripts/verify/k3_chat.sh` exit 0，纯 CPU 本机）**。crate 侧 `models/kimik3/src/{tokenizer,xtml,parser}.rs`，
> 对拍 `models/kimik3/tests/chat_parity.rs`，golden `test/kimik3/golden/chat.json`（393 KB，进仓库）由
> `scripts/verify/gen_k3_chat_golden.py` 生成（只用模型目录的 `tokenization_kimi.py` + `encoding_k3.py`，不需要 torch）。
> 下一步 C-4b：`service.rs`（`ChatBackend`）+ `executor.rs`（接 `dinfer_layers::exec` 公共件）。

## 环境

- 参考：`/data/ref/models/kimik3/{tiktoken.model,tokenizer_config.json,generation_config.json,tokenization_kimi.py,encoding_k3.py}`；
  vLLM 基线 `/data/ref/vllm`（`de8291e`）的 `renderers/kimi_k3.py`、`reasoning/kimi_k3_reasoning_parser.py`、`tool_parsers/kimi_k3_tool_parser.py`。
- 用户指定的对齐经验来源：`/data/code/llm-cache-prediction`（只读；`docs/kimik3/{README,other}.md`、`tools/gen_goldens.py`、`testdata/golden/cases.json`）。
- python：`/data/ref/venv-tf519/bin/python -I`（transformers 5.19 / tiktoken 0.9），本机 CPU 即可。

## 观察方法 / 停止条件

`scripts/verify/k3_chat.sh`（exit 0 = 过）= 生成 golden + `cargo test -p dinfer-kimik3`：
1. `specials_and_eos_match`：256 特殊 token 名 ↔ id、eos；
2. `encode_matches_python_both_modes`：12 段文本 × `allow_special ∈ {true,false}` ids 逐位一致 + decode 往返；
3. `incremental_decode_matches_prefix_decode`：逐 token 增量 detokenize 与 Python 逐前缀 decode 一致（半个 UTF-8 不吐）；
4. `chat_render_matches_python_both_dialects`：35 个 case × 2 方言 = 70 条，**文本与 ids 逐位一致**（含 2 条预期报错）；
5. `parser_matches_rendered_messages_whole_token_and_char`：7 条"模型输出"按整段 / 逐字符 / 逐 token（经增量 detokenize）三种喂法，
   reasoning / content / tool_calls（name + arguments JSON 值）三者一致。
迭代上限 10 轮；本次第 1 轮全过（golden 脚本自身改了 3 次：tool 消息名解析、parser case 的前缀切分）。

## 结论（与 HF / vLLM 核对后的事实）

### tokenizer（`tokenizer.rs`）
- ranks 163584 条 → 特殊 token 空间 `[163584, 163840)`；有名 16 个（表见 `design-kimik3.md` §1.5），其余 `<|reserved_token_{i}|>`。
- `pat_str` 逐字照抄；**用 `fancy-regex 0.14`**（tiktoken 自身就是 `fancy_regex`：类交集 `&&`、`(?!\S)` 语义一致）。
  踩坑：Python `regex` 模块默认（V0）**不认 `&&`**，把它当字面字符，拿它当预分词参照会得到错误切分（`" world"` 被吞）。
- `allow_special=true` = `allowed_special="all"`：先按 256 个名字切，中间段普通 BPE；`false` = `disallowed_special=()`：
  整段普通 BPE，用户文本里 `<|open|>` → `<|` `open` `|>` 三个普通 token，绝不变控制 token。
- 长文本切分复刻：400k 字符窗口 → 连续（非）空白 run > 25k 字符处切 → 每段独立 BPE（`long_ws_run_split` / `long_nonws_run_split`
  两个 case 覆盖 26000 空格、25010 个 x；切点确实改变 BPE 结果，不复刻就差 1–2 token）。Python `str.isspace()` 比 Rust
  `char::is_whitespace` 多 `\x1c–\x1f`，已补。
- BOS 不自动前置；eos = `<|end_of_msg|>`（163586，`generation_config.json`），`[EOS]` 163585 不是生成终止符。
- 增量 detokenize：bytes 缓冲 + `from_utf8().valid_up_to()`，非法序列（非"未完"）整体 lossy 不再等。

### XTML 渲染（`xtml.rs`）
- `build_chat_segments` 逐函数移植；`thinking_effort` 默认 `max`（`apply_chat_template` 的 `setdefault`，None 显式传入才不注入）；
  非法值（`medium`）**报错**（HF assert / vLLM 400），不像 GLM 那样回退 max。`reasoning_effort="none"` → thinking=false
  （vLLM `_apply_k3_thinking_kwargs`）。
- tool 消息名解析顺序：`tool` / `name` 字段 → 最近 assistant `tool_calls` 按序；都没有 → 报错（HF 原文）。
  `normalize_xtml_tool_result_messages`：按 `tool_call_id` 匹配重排 + 覆盖 `tool`/`name`；一条匹配不上整段原样。
- **两种方言 `Dialect::{Hf, Serving}`**（见下"采纳 / 未采纳"）：默认 `Hf`。

### 解析器（`parser.rs`）
- 状态机 Think → AfterThink → Response → AfterResponse → Tools → Done；初始 Think（生成提示以 `<|open|>think<|sep|>` 结尾），
  模型若重复输出 open 标记则吞掉；`thinking=false` 时初始 Response。
- 标记跨 token：尾部是任一标记真前缀就 hold back（vLLM `_partial_tag_overlap`）；工具调用整块闭合才吐（vLLM 同）；
  `type="string"` 原文、其他 JSON 解码失败退原文、属性 `&quot;`/`&amp;` 反转义（vLLM `_decode_call` 同）。
- 与 vLLM 的差别：vLLM 无 tools 的请求只走 reasoning parser 并剥 response 包裹；我们单状态机恒解析 tools 段。
  额外支持 `<|open|>json type="object"<|sep|>…` 块（渲染侧畸形参数会这样写，vLLM parser 不认）。

## 采纳 / 未采纳（来自 `/data/code/llm-cache-prediction`，用户 2026-10-20 要求记录）

| 它的结论 | 采纳？ | 怎么做的 |
|---|---|---|
| ① 渲染参照物是 **Moonshot 线上 vLLM 的 Rust renderer**，不是 HF `encoding_k3.py`：声明 / schema 路径 pydantic 塌整数浮点（`100.0`→`100`、`1e16`→`10000000000000000`、`1e-5`→`0.00001`、`1e-6` 保留）；tool_calls 字串参数 serde 完整解析重序列化（`[1,  2]`→`[1,2]`、`1e2`→`100.0`） | **部分**：做成 `Dialect::Serving`，golden 脚本按它的 monkeypatch 方式（自写代码）生成 serving 口径；**默认用 `Dialect::Hf`** | 我们的分析基线 vLLM `de8291e` 的 `renderers/kimi_k3.py` 直接调 HF `apply_chat_template`（Python），没有 Rust renderer（`/data/ref/vllm/rust` 不含 chat）；基线 vLLM 的 token 序列 = HF 口径。Serving 口径的真值是它在线探针，本机无法复核，所以不做默认。服务起来后可配置切换。 |
| ② tiktoken：special 空间、`<|reserved_token_{i}|>`、`allow_special` 分段、**400k/25k 切分** | ✅ 全部 | `tokenizer.rs`，golden 有两个长文本 case 覆盖切点 |
| ③ 第 3 处偏差：消息 `name` 被 vLLM 前端丢弃 | **按方言**：`Serving` 不渲染；`Hf` 渲染 | 基线 vLLM `chat_utils.py:2158` **保留** `name` 并传给 HF 渲染，与它的部署不同 |
| ③ 第 4 处偏差：`thinking` 可以是对象 `{effort, keep, type}` | 记录，接口不写死 | `RenderOptions{thinking, thinking_effort}` 由 service 层从 `reasoning_effort` / `chat_template_kwargs.thinking(.effort)` 映射（C-4b） |
| ③ 图片占位从**所有角色**抽（含 tool 消息） | 接口已留 | `render_content_segments` 对所有角色处理 `image`/`image_url` part → `<|kimi_image_placeholder|>`（allow_special 段）；视觉塔接入前只是文本 |
| ④ case 清单（G01–G20、`cases.json` 28 条）| ✅ 作为覆盖清单 | 本 golden 35 case：单/多轮、三档 effort + none + 非法、tools 声明（浮点 / unicode / 未排序 key / 动态 system tools）、dict / 字串 / 畸形 / 空参数、tool 结果重排 / 不匹配 / 按序、tool_choice、response_format ×2、name 属性转义、text parts、特殊 token 字样、CJK+emoji+ZWJ、代码块空白、长 run、缩写与数字。非 ASCII 函数名被 vLLM 400 是端点校验，渲染层不拦 |
| ⑥ 命中粒度 1536 = 12×128（`--mamba-cache-mode align`） | 顺带发现，已在 `design-kimik3.md` §1.6 | 不属于 C-4a |

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 1 | 三个模块 + golden 脚本 + 对拍测试 | 单测 19/19（含 kimik3 既有 11 个）、chat_parity 5/5：70 chat case、24 encode、24 增量、7 parser × 3 喂法 全过 |

归档：`test/kimik3/golden/chat.json`（golden 本体）；日志 `/tmp/k3_chat.log`。
