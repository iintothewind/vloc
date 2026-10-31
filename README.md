# vloc

把 [WeVisOne-VL](https://github.com/Tencent/WeVisOne-VL)（腾讯 WeChat Vision，Apache-2.0）的感知能力封装成供 agent 调用的 Web API。覆盖检测、指代定位、计数、指向、OCR、分割和 alpha matting，共 11 个任务。

当前阶段：能力调研、选型和压测已完成，推理镜像已经容器化。API 服务本身还没写，落地顺序见报告 §9。

## 目录

| 路径 | 内容 |
|---|---|
| [`docs/wevisone-vl-api-report.md`](docs/wevisone-vl-api-report.md) | 能力调研与 API 设计报告：能力清单、输入输出契约、实测性能、选型（2B BF16 vs 4B Q4_K_M）、同步/异步批量 API、输入验证、大图缩放、容器化 |
| `docker/` | 推理镜像：`Dockerfile`、运行时最小依赖 `requirements.txt`、GPU 冒烟脚本 `smoke.py` |
| `bench/` | 压测和评测脚本，在 GPU 主机的工作目录下运行（见下文） |
| `results/` | 原始测量结果（JSONL / 日志 / `arms.tgz`），报告中的数字都出自这里 |
| `.github/workflows/docker.yml` | 构建 amd64 + arm64 镜像并推送到 GHCR |
| `AGENTS.md` | 协作红线：**工作站上禁止运行任何模型** |

## 结论速览

- **模型选型：2B BF16**。在同一像素预算下，它的检测 F1 与 4B Q4_K_M 持平，拒答更好，显存相同（约 4.7 GiB），并且保留 mask/alpha 输出。GGUF 路径没有像素任务
- **显存**：2B 全能力峰值 11.9 GiB（2 MP 输入），12 GB 显卡能放下；4B 全能力需要 24 GiB 级显卡
- **吞吐**：单进程 2B 约 0.78 rps；`predict_batch` 一次 16 张图时，box 任务降到每张 0.23 s
- **输入**：服务端先等比缩放到约 2 MP（mask IoU ≥ 0.9993），检测 label ≤ 20，`max_new_tokens` ≤ 1024

详细数据和依据见报告 §1 与 §12。

## Docker 镜像

```
ghcr.io/iintothewind/vloc:latest     # main 分支
ghcr.io/iintothewind/vloc:sha-<7>    # 每次提交
ghcr.io/iintothewind/vloc:<x.y.z>    # 推送 v* tag 时
```

多架构清单，包含 `linux/amd64` 和 `linux/arm64`。镜像内含 Python 3.12、`torch 2.11.0+cu130` 和钉死 commit 的 `wevisone_vl` 引擎，**不含权重**。

宿主机要求：NVIDIA 驱动 ≥ 580，并安装 nvidia-container-toolkit。

```bash
# 权重在宿主机上拉取（git-lfs）
git clone --depth 1 https://huggingface.co/tencent/WeVisOne-VL-2B models/WeVisOne-VL-2B

# GPU 冒烟：加载模型，对 /smoke/dogs.jpg 各跑一次 detection 和 segmentation，输出 JSON
docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/images:/smoke:ro" \
  -e WEVISONE_MODEL=/models/WeVisOne-VL-2B \
  ghcr.io/iintothewind/vloc:latest
```

本地构建：

```bash
docker build -t vloc:dev docker/
# 驱动较老的主机可以改用 cu128 轮子
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t vloc:cu128 docker/
```

CI 在两个架构的原生 runner 上分别构建（`ubuntu-24.04` 和 `ubuntu-24.04-arm`，不经过 QEMU），按 digest 推送后合并成一个 manifest。触发条件是 `main` 上 `docker/**` 或工作流文件有改动、推送 `v*` tag，也可以手动 `workflow_dispatch`。

## 压测脚本

所有模型推理只在 GPU 主机上执行。脚本用 `scp` 拷到 GPU 主机的工作目录，在那里运行，结果再取回本仓库的 `results/`。工作目录需要包含：`WeVisOne-VL/`（官方仓库 clone）、`models/WeVisOne-VL-{2B,4B}/`、已装好引擎的 `venv/`。

| 脚本 | 用途 | 产物 |
|---|---|---|
| `bench/bench.py <reps> <model_dir>...` | 稳态延迟、顺序队列吞吐、批处理 vs 逐张、双引擎显存 | `results/bench_results.jsonl` |
| `bench/limits.py <model_dir>` | 输入尺寸、批大小、label 数、query 长度扫描 | `results/limits_2B.jsonl` |
| `bench/coco_prep.py` | 生成 COCO val2017 的 60 图子集（固定种子） | `coco/subset.jsonl` |
| `bench/cmp_arms.py native\|llama ...` | 单个对比臂的原始预测 | `arms/*.jsonl` |
| `bench/score_arms.py <arm.jsonl>...` | box F1@0.5、F1 均值（IoU 0.5:0.95）、拒答率 | stdout |
| `bench/run_arms.sh` | 2B/4B BF16 与 4B Q4_K_M（打补丁的 llama.cpp）全流程对比 | `results/arms.tgz` |

## 许可

WeVisOne-VL 的代码和权重遵循 Apache-2.0（见上游仓库）。COCO 标注遵循 CC BY 4.0。
