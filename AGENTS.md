# agents.md

## 绝对红线：禁止在本机运行 LLM

**绝对不能在本机（这台 Windows 工作站）运行任何 LLM / VLM 推理。** 这条是硬性约束，任何任务、任何理由都不得违反。

禁止在本机做的事：

1. **运行模型推理** —— 加载权重、`model.generate()`、加载 safetensors/GGUF、跑探针脚本、起 Gradio/vLLM/Ollama/llama.cpp/LM Studio 等任何会占用 GPU 显存去跑大模型的进程。本会话曾因此把机器搞死机。
2. **安装模型运行栈** —— 不得在本机 pip 安装 `torch` / `torchvision`（尤其 CUDA wheel）/ `transformers` / `accelerate` / `vllm` / `triton` / `flash-attn` / `qwen_vl_utils` 等推理依赖，也不得为它们创建专用 venv。
3. **把权重落到本机** —— 不得 `git clone` HuggingFace 模型仓库、不得 `hf download`、不得让 `~/.cache/huggingface` 增长。GB 级权重文件属于远程主机。
4. **用本机 GPU 做压测 / 显存峰值测量** —— 这类测量全部在远程主机上做。
5. **占 GPU 的附带工作** —— 本机 GPU 显存已被现有服务占满（RTX 4070 12282 MiB，实测 11764 MiB 在用 [M]），任何本地 GPU 作业都会与之争抢。

模型相关的一切（权重、依赖、推理、压测）只在远程主机上执行，通过 SSH 操作，产物（日志、JSON 测量结果）取回本机即可。本机允许的工作：纯 CPU 的解析器 / API 服务 / 测试脚本 / 文档，以及读写代码。
原因：本机已经在运行 LLM（`code/strata` 的 server，22:07 起，port 8888，占 GPU 显存近满），本机资源不足以再跑模型。用户在本会话多次明确指令，逐字记录：
- 「等一下不要尝试本地运行, 本地运行着llm, 你这样做会让本机崩溃」（02:33）
- 「本地运行不了, 没资源!」（03:32）
- 「ssh spark2 你可以再spark2 的 ~/Documents/vlocate 目录 进行允许调试, spark2 有足够的uma使用. 测试然后回答 …」（03:42）

**唯一的测试/调试主机是 spark2**（`ssh spark2` = `acro@10.66.0.12` 免密，已存本地 ssh 配置；DGX Spark GB10，128 GB 统一内存）。工作目录固定 `~/Documents/vlocate`，所有权重、venv、探针脚本、压测都放那里，跑完把 JSON/日志取回本机。

If a task requires running a model, report the constraint and run it remotely — never locally.
