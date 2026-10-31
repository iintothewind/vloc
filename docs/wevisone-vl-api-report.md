# WeVisOne-VL 能力封装报告：面向 Agent 的 Web API

生成日期：2026-10-29 · 仓库：<https://github.com/iintothewind/vloc>（本报告位于 `docs/`，探针脚本在 `bench/`，原始结果在 `results/`，镜像定义在 `docker/`）

事实标注约定：
- **[M]** = 本会话/一手来源直接实测或读码确认
- **[3P]** = 第三方实测（引用出处）
- **[I]** = 由模型配置与硬件规格推算

---

## 1. 结论摘要

WeVisOne-VL（腾讯 WeChat Vision，Apache-2.0，基于 Qwen3-VL）把检测、指代定位、计数、指向、OCR、指代分割、alpha matting 全部统一成结构化 next-token 预测：坐标 token `<0>`…`<1000>`、`None` 表示目标不存在、`<OBJ>` 标记像素级输出。像素级输出（mask / alpha）由一个独立的 **9.6M 参数 pixel decoder**（`mask_decoder.pt` sidecar，143 MB [M]）从 backbone 最后一层 hidden state + 视觉中层特征解码，**不需要 SAM**。

对封装成 API 的核心结论：

1. **能力覆盖度取决于部署路径**：只有 `transformers` + `WeVisOneVL` 引擎能产出 mask/alpha；vLLM、Ollama/GGUF、llama.cpp 只能出 box/point/OCR 文本结果。
2. **官方引擎的 `boxes` 字段只填像素任务**（`parse_seg_boxes` 只解析紧跟 `<OBJ>` 的 box）。纯 box/point 任务的真值只在 `response` 原始文本里 → **服务端必须自己写全量 `<box>` 解析器**。
3. **无置信度、无 polygon、无视频、无多图（引擎内）、无 CoT**；坐标量化精度 1/1000。
4. **本机不跑模型（红线，见 `AGENTS.md`）**：本会话在这台工作站（RTX 4070 12 GB / 48 GB RAM [M]）装 CUDA torch + 2B 权重跑推理导致死机，本地安装已全部清除。运行主机只能是服务器/远程 GPU（本会话全部实测数据来自 spark2 GB10）。显存预算按稳态压测口径 [M]：**2B 全能力 11.9 GiB @2 MP**（1.75 MP 单 mask 7.5 GiB，box-only 4.9 GiB）→ 12 GB 卡只放 2B；**4B 全能力 21.1 GiB @2 MP**（16.8 GiB @1.75 MP，box-only 14.2 GiB）→ 需 24 GiB 级；2B+4B 双引擎常驻 **20.5 GiB**；4B 量化（Q4_K_M 4.38 GiB [3P]）**丢 mask 能力（官方确认）**，只能做 box-only
5. GB10 稳态实测（6 重复/任务 [M]）：2B p50 **point 0.56 / referring 0.77 / detection 0.97 / count_box(16) 3.00 / ocr 3.93 / 像素任务 1.24–2.68 s**；4B p50 **1.25–15.3 s（box）/ 3.97–6.54 s（像素）**。容量规划用吞吐而不是延迟：单进程常驻 2B **0.78 rps**、4B **0.217 rps**（12 请求混合队列实测），`predict_batch` 给 box-only 端点 **3.2–3.8× 增益（2B 0.24 s/图）**，冷启动 **load 34.4 s (2B) / 67.3 s (4B)** 必须算进 SLA
6. **选型结论（§12.1，spark2 COCO 60 图配对实测 [M]）：选 2B BF16，不选 4B Q4_K_M**。在同一像素预算下，Q4 的 box F1@0.5 与 2B 打平（79.7 vs 79.6），F1 mean 高 2.3，但拒答率更低（96.7% vs 98.3%）、**没有 mask/alpha**、显存几乎相同（Q4 进程 4.64–4.81 GiB，2B box-only 峰值 4.73 GiB），而且需要第二套运行时（打补丁的 llama.cpp）。服务形态：同步 `/v1/perceive` 加异步 `/v1/batches`；服务端先等比缩放到 `max_pixels`；`max_new_tokens` 上限 1024；容器镜像已在 spark2 构建并用 GPU 冒烟通过（§12.2–12.5）

---

## 2. 能力清单

`wevisone_vl/prompts.py` 的 `TASKS` 是权威任务表（11 项，逐字读码确认 [M]）。`WeVisOneVL.predict(image, task=…, query=…, reference_boxes=…)` 按 `build_prompt(task, query, …)` 生成 canonical prompt：

| task key | 能力 | Canonical prompt（逐字 [M]） | 输出 |
|---|---|---|---|
`detection` | 标签列表 → 所有实例 box（开词表密集检测） | `Locate all the instances that match the following description: {labels joined by ". "}.` | box + 标签 |
`referring` | 指代表达式 → box | `Locate the instances that match the following description: {phrase}.` | box |
`point` | 标签列表 → 点 | `point to: {labels joined by ". "}` | 点（2 token） |
`count_box` | 计数 + 逐实例 box | `How many {query} in the image? Please count them with boxes.` | 数字 + boxes |
`count_point` | 计数 + points | `How many {query} in the image?` | 数字 + points |
`segmentation` | 指代分割（二值 mask） | `Locate the instances that match the following description and provide the alpha mattes: {phrase}.` | box + `<OBJ>` → mask |
`matting` | 文本提示 alpha matte（发丝/半透明边缘） | 同 `segmentation` | box + `<OBJ>` → alpha |
`ocr` | 全图文字定位+识别 | `detect all the word in the box format` | box + 文本 |
`ocr_grounding` | 指定文本串 → 位置 | `Locate the following text in the image: {text}` | box |
`visual_prompt` | 参考 box → 找全图相似物体（demo 未暴露） | `Locate all similar objects in the image based on the reference boxes: <ref>object 1</ref><box>{tokens}</box>`，`reference_boxes` 为**像素 xyxy**，经 `_box_tokens` 归一到 `<0>`–`<1000>` [M] | boxes |
`freeform` | 任意 query 直答（grounded caption / VQA，无 CoT） | query 原样透传 | 文本 |

模型级附加能力：**目标不存在时输出 `None`**，WeRef-Bench 拒答率 96.4%（基线 Qwen3-VL-8B 18.3%）[M, 官方 README 表]。API 必须把拒答显式化，且**粒度不同**：referring 是整条 query 级（`<ref>{phrase}</ref><box>none</box>`），dense detection 是逐 label 级（`<ref>people</ref><box>none</box>`，demo parser 对该 label 做 `predictions.pop`，即缺失 label 在输出里直接消失）→ schema 用逐 label `absent` 字段，不让 agent 解析文本。

HF 空间 demo（2B/4B）用 10 个 task label + 7 个 display mode（`Auto` + segmentation 组 `Overlay / Binary Mask / Cutout` + matting 组 `Green Screen / Overlay / Alpha Matte / Cutout`，逐字 [M]），经 `gradio_client` 实测确认 demo 有两个端点：`/on_task_change`（task → canonical query 字符串）与 `/run(image, task_label, query, max_new_tokens, temperature, display_mode)` [M]。demo 的 task label 与 `TASKS` 的映射：Grounding→`referring`，Dense Detection→`detection`，Counting (boxes)→`count_box`，Counting (points)→`count_point`，Pointing→`point`，OCR (all text)→`ocr`，OCR (text grounding)→`ocr_grounding`，Grounded Caption/VQA→`freeform`，Segmentation→`segmentation`，Matting→`matting`。

### 四个 box 族的选择规则（spark2 实测语义差异 [M]）

| 需求 | task | 实测逐字行为依据 |
|---|---|---|
| 描述→位置，带语义条件、可拒答 | `referring` | `boys.jpg` + `boys holding microphone` → 6 个男孩中只框**符合"手持麦克风"的 2 个**（语义过滤，2B/4B 一致）；`the chair near the window`（不存在）→ `none` |
| 类别清单全量列出、逐标签可枚举缺失 | `detection` | `boys\nmicrophone` → 6 boys + 2 (2B) / 3 (4B) microphone，**全出、不做语义过滤**；缺失 label 单独输出 `none` |
| 要数量答案 | `count_box`（实例 ≤20，box 可复用于裁剪/追踪）/ `count_point`（密集目标，2 token/实例，实测快约 40%） | `chess.jpg` + `white chess pieces` → 16 boxes + `Total number: 16.` |
| 要坐标不要数字 | `point` | `point to: dogs` → 4 points，**无 `Total number` 行**（与 `count_point` 的唯一差别就是计数声明） |

### 官方基准（4B，选摘 [M]）

| 任务 | 基准 | 指标 | 4B | 2B | 最强基线 |
|---|---|---|---|---|---|
| Detection | LVIS | F1 Mean | **55.3** | 53.3 | 50.7 |
| Dense Detection | Dense200 | F1 Mean | **71.3** | 61.4 | 58.7 |
| Referring | RefCOCOg test | F1 Mean | **81.8** | 78.2 | 77.6 |
| Rejection | WeRef-Bench | Rej. Rate | **96.4** | 92.9 | 73.0 |
| Counting | PixMo-Count | Acc | **89.6** | 88.1 | 82.8 |
| Pointing | PointBench | Acc | **71.9** | 62.9 | 69.9 |
| OCR | TotalText | F1 Mean | **52.0** | 42.5 | 43.3 |
| Referring Seg | ReasonSeg val | gIoU | **71.2** | 60.9 | 65.0 |
| Box Matting | P3M-500-NP | MAD×10³↓ | 7.3 | **6.6** | 15.4 |
| Text Matting | WeRefMat | MAD×10³↓ | **9.8** | 15.8 | 22.4 |

评测口径：贪心解码，每图 ≤1,960 visual tokens [M]。

---

## 3. 输入契约

`WeVisOneVL.__init__` / `predict` 的读码事实 [M]：

- **图像输入**：`PIL.Image` 或路径（`Image.open`）；`predict` 立即 `convert("RGB")` → **RGBA 的 alpha 被丢弃**，透明图必须客户端先合成背景
- **像素预算**：`min_pixels = 16*32*32 = 16,384`，`max_pixels = 2025*32*32 = 2,073,600`（≈2 MP）；processor 自动 resize，alpha 输出再 resize 回原尺寸
- **视觉 token 数学**：vision patch 16 × `spatial_merge_size` 2 → **1 visual token = 32×32 = 1,024 像素区域**；`max_pixels` 上限对应 **2,025** visual token，官方评测口径是 **≤1,960**（即评测时像素预算被压到 ≈2.0 MP 以内）；`min_pixels` → 16 token
- **1080p 正好在预算边界**：`1920×1080 = 2,073,600 = max_pixels`（精确相等 [M]）。smart_resize 取 28 的倍数：1920→1904、1080→1064，得 2,021,056 px = **1,974 visual token**，已略超评测口径 1,960 → 1080p 输入要拿到 1,960 token 需显式调低 `max_pixels`
- **单图单轮**：引擎 `predict` 只接一张图；vLLM backend `limit_mm_per_prompt={"image": 10, "video": 10}`（引擎内无多图任务）
- **上下文**：模型 `max_position_embeddings = 262,144`（4B config [M]）；vLLM 路径 `max_model_len` 默认 **8,192**，`gpu_memory_utilization` 默认 0.45
- **生成默认**：`max_new_tokens=4096, temperature=0`（贪心，`do_sample=False`），`top_p=0.95, top_k=50, repetition_penalty=1.05`，stop `<|im_end|>`
- **响应长度**：单条响应 ≤500 box（训练上限）；demo slider 上限 1024/2048 token
- **EXIF 方向不会被自动应用**（LiteRT 转换卡明确记录，且预处理链就是 resize-to-multiple-of-28）→ 服务端必须自己做 `ImageOps.exif_transpose`
- prompt 是固定模板族：`referring`/`matting` 的 query 是**单短语**（`_phrase` 补句点），`detection`/`point` 的 query 是**换行分隔标签列表**（`_labels`），`ocr_grounding` 用空格拼接（`_text`）—— 混用会破坏训练分布
- `predict(prompt=…)` **可覆盖模板**（`prompt` 参数逐字 [M]）→ 生产 API 必须只暴露 `build_prompt` 生成的 canonical prompt，不开放 raw prompt 注入
- **`reference_boxes` 只在 `visual_prompt` 分支被消费**（`build_prompt` 逐字 [M]：只有 `visual_prompt` 用 `_box_tokens(reference_boxes, image_size)`）→ 把 box 传给 `matting`/`segmentation`/`detection` 会被**静默忽略**，query 变成字面短语（训练分布外）。README 案例研究里的 "Box-prompted Matting" 在 released prompt 表里**没有模板**

## 4. 输出契约

`InferenceResult`（dataclass [M]）：`response`（raw 文本）、`prompt`（实际模板）、`boxes`（归一化 xyxy 0–1）、`masks`（`float32` 概率图，`F.interpolate` bilinear 到输入图 H×W）、`latency_s`（generation + 可选 mask decode 的墙钟）。

关键陷阱（逐条读码确认 [M]）：

1. `boxes`/`masks` **只在响应含 `<OBJ>` 时填充**（`predict` 里 `if decode_matting and SEG_RE.search(response)`）。`detection`/`referring`/`point`/`count_*`/`ocr` 的 `boxes` 是空列表。
2. **响应语法是 label 块，不是逐框**（`demo/app.py:614` `parse_predictions_with_masks` 逐字 [M]）：`<ref>label</ref><box>x1 y1 x2 y2, x1 y1 x2 y2, …></box>` 与 `<ref>label</ref><point>x y, x y, …></point>` —— **一个 label 一个块，块内逗号分隔多实例**。demo 正则 `<ref>(.*?)</ref>\s*<(box|point)>(.*?)</\2>`（闭合靠 backreference），body 按 `,` 切段，`step = 2 if point else 4`，坐标直接 `/1000 × width/height`。engine 的 `parse_seg_boxes` 只服务 `<OBJ>` 像素任务，且每段只取**最后一个** 4 元组 —— 对 box-only 任务它只得到最后一个 box（spark2 实测：16 实例响应的 `result.boxes` 只有 1 组 [M]）。
3. box = 4 坐标 token + `<OBJ>`（像素任务）；点 = 2 坐标 token。
4. **无置信度分数**；无 polygon 输出（mask 是逐对象概率图，不是多边形）。
5. 坐标量化 = 1/1000 相对宽/高：1920 px 宽图 ≈1.92 px，4000 px 宽图 ≈4 px 的固有误差。
6. mask decode 逐 box 可能失败（`decode_masks` 里 `count = min(len(boxes), t_seg.shape[0])`，`n_seg <= 0` 抛 `RuntimeError`）→ API 需要 per-object 状态字段。
7. `count_text` 在 engine 中未实现（`NotImplementedError` [M]）。
8. **`decode_masks` 的 `boxes` 参数是归一化 xyxy 覆盖**（docstring 逐字："overrides the boxes parsed from `response`, e.g. to condition the decoder on unrounded ground-truth boxes" [M]）→ **box-prompted matting 可在服务端合成实现**：用真实 `segmentation` 模板做 teacher-forcing prompt，配一条合成响应 `<ref>dog</ref><box><x1><y1><x2><y2><OBJ>`，把客户端 box 作为 `boxes` override 传入（spark2 实跑验证见 §5.1）

### 响应语法的三处实测行为（spark2 GB10 原生引擎，逐字 raw response [M]）

- **per-label 拒答**：dense detection 的缺失标签输出 `<ref>people</ref><box>none</box>`；referring 的拒答是整条 query 级 `<ref>{phrase}</ref><box>none</box>`。API 的 `target_absent` 必须是逐 label 字段，不是全局布尔。demo parser 对空 bucket 做 `predictions.pop(cat)`，即**缺失 label 在 UI 输出里直接消失**。
- **显式计数行**：`count_box`/`count_point` 在实例块后追加 `\nTotal number: N.`。实测 4B 在 `chess.jpg count_point` 声明 15 且吐出 15 个 point，2B 是 16 并与 `count_box` 的 16 boxes 逐字对应 → **模型自己会数错**，API 必须同时返回 `declared_total` 与 `count_by_items` 并标注不一致。
- **畸形闭合**：4B 在 `livingroom.jpg count_box "chairs"` 输出 `<ref>chairs</ref><point>none</box>\nTotal number: 0.`（开 `<point>` 关 `</box>`），backreference 匹配不到 → label 丢失；同一场景 4B `count_point` 吐 **31 points**（2B 为 1）。密集场景的过度计数与闭合畸形是实测行为 → 解析器需允许 box/point 任意闭合配对，并对 count 做数字↔实例数交叉校验。

### Pixel decoder 内部结构（README + mask_decoder.py [M]）

- 输入图 resize 到 4 的倍数，grid `S = image size × 4`；轻量 CNN 把 S 降到 1024；1024 网格上的 fusion 特征池化到 512；deformable cross-attention 块把 fusion 特征聚合成 mask token；cascade deformable conv 从 S/4 → S/2 → S/1 逐级重建，最终 alpha resize 回输入尺寸
- 需要 backbone **最后一层 hidden state**（`<OBJ>` 位置的 `t_seg`）+ `cfg.vision_feature_layers` 的 premerge 视觉特征，通过 `LastHiddenCapture` / `VisualFeatureCapture` hook 捕获；teacher-forcing 重跑整个序列且 `use_cache=False`（即**每个像素任务多一次全量 forward**）
- `decoder_batch_size` 默认 16；`attention_implementation` 默认 **`sdpa`**（README 说 flash-attn 为 pixel 任务首选、sdpa 为 fallback）→ Windows 无 flash-attn 可用 sdpa 跑通
- sidecar 解析：本地目录需含 `mask_decoder.pt`，否则 `FileNotFoundError`；远端走 `hf_hub_download`

### 模型配置事实

4B [M, config.json]：`Qwen3VLForConditionalGeneration`，bf16，text 36 层 / hidden 2560 / heads 32（KV 8）/ head_dim 128 / intermediate 9728，vocab 152,680（含 1,001 个坐标 token，`<OBJ>` = 152,678 [3P, GGUF 校验]），vision depth 24 / hidden 1024 / patch 16 / merge 2 / deepstack `[5,11,17]`，`max_position_embeddings = 262144`。权重单文件 `model.safetensors` **9.66 GiB**（4.83B 参数）[M]，`mask_decoder.pt` 143 MB [M]。2B：repo 4.93 GB [M]，28 层 / hidden 2048（Qwen3-VL-2B 基座）[I]。

---

## 5. 实测性能

### 5.1 已实测数据

| 环境 | 任务/规模 | 结果 | 来源 |
|---|---|---|---|
| HF demo 4B / zero-A10G | Grounding 2 实例 | **服务端 1.5 s**（`result.latency_s`），端到端 2.7–8.1 s 含上传排队；匿名 IP 约 **7 次请求**后 ZeroGPU 配额耗尽 | 本会话实测 |
| PALLER-NOVUS pilot（RTX 2060 6 GB，context 4096，image-token 1024–1960，贪心，2048 输出上限，单 worker） | 20 query 均值 | 2B Q4_K_M **3.44 s / 2.91 GiB**（19/20 可解析）；2B F16 5.20 s / 5.10 GiB；4B Q4_K_M **4.17 s / 4.38 GiB**（20/20）；4B F16 59.90 s / 5.29 GiB（CPU+GPU offload，非纯 GPU） | 第三方实测 |
| LiteRT-LM 2B f16，Galaxy S26 GPU | 512×512 问答 | **TTFT 1.53 s（其中视觉编码 0.90 s），decode 12.97 tok/s**，峰值 6.6 GB | 第三方实测 |
| LiteRT-LM 2B f16，Apple M4 Max GPU | 同上 | **TTFT 0.30 s，81.4 tok/s**；CPU：TTFT 4.85 s，12.6 tok/s；峰值 51.5 GB | 第三方实测 |
| LiteRT Android app（含 pixel decoder，matting 全路径） | 640×480 / 3000×2029 | **4.7 s / 5.8 s**，首次 35.7 s（warmup） | 第三方实测 |
| GGUF 4B Q4_K_M + projector | 文件体积 | ≈3.2 GB，Ollama 一行 pull；**mask 能力丢失（官方确认）** | 第三方 [3P] |
| spark2（DGX Spark GB10，128 GB 统一内存，原生 transformers 引擎 bf16，贪心，`result.latency_s`） | 18 请求，2B/4B 全 box 任务 | 2B **0.32–3.0 s**，4B **0.57–5.7 s**；16 boxes 的 `count_box` 最贵（2B 2.99 / 4B 5.66 s），16 points 的 `count_point`（2.14 / 3.79 s），referring 拒答最快（0.37 / 0.67 s）；warmup 2.0–2.4 s；模型侧 = 墙钟（无排队） | 本会话实测 |
| spark2 2B（像素任务，`dogs.jpg` 1024×696 = 0.71 MP，4 个 `<OBJ>`） | `segmentation` 端到端 | **2.51 s**（generation + teacher-forcing 全量 forward + 4 mask 解码），峰值 **11,292 MiB**，mask shape **= 输入尺寸** `696×1024`、`float32`，逐对象 min 0 / max 1、mean 0.063–0.133、`frac>0.5` ≈ mean（二值性良好） | 本会话实测 |
| spark2 4B（同上，`dogs.jpg` 0.71 MP，4 个 `<OBJ>`） | `segmentation` 端到端 | **8.569 s（冷启动：该模型会话首请求，含 pixel decoder warmup）**，峰值 **15,898 MiB（≈15.5 GiB）**；预热后稳态参考 = 4B @2 MP 的 **6.106 s**（§5.1 的 2 MP 行）；逐对象 mask stats 与 2B 几乎一致（mean 差 ≤0.001） | 本会话实测 |
| spark2 `visual_prompt`（参考 box = 第一只狗的像素 xyxy） | 2B / 4B | **2.23 s / 3.69 s**；响应 `<ref>object 1: Dog</ref><box>…4 boxes…` —— **模型给参考目标命名并全图找同类**；`boxes` 为空（响应无 `<OBJ>`），mask 不触发 | 本会话实测 |
| spark2 `matting_box` 合成路径（`decode_masks(..., boxes=[client_box])` + 单 `<OBJ>` 合成响应，仅解码） | 2B / 4B | **0.927 s / 0.820 s**；输出 mask 与该对象 segmentation 结果几乎逐值一致（mean 0.1324 vs 0.1326 / 0.1332 vs 0.1333）→ **box-prompted matting 的封装路径实测成立** | 本会话实测 |
| spark2 像素任务 @1.75 MP（`seabreeze.jpg` 1620×1080） | `matting`（`boat` / `person`） | 2B **1.392 s / 1.219 s**，峰值 **7,515 MiB**；4B **1.983 s / 1.822 s**，峰值 **12,126 MiB**；alpha shape = 输入尺寸 `1080×1620`、`float32`、min 0 / max 1，mean 0.075（boat）/ 0.272（person）；8-bit PNG 含全部 256 级，落在 (0.05, 0.95) 的像素占 **2.87%（2B person）/ 3.23%（4B person）** → **alpha 是连续软边，服务端不得二值化** | 本会话实测 |
| spark2 @2 MP（1920×1080） | `segmentation` 4 对象 / `detection` | 2B **2.462 s / 1.683 s**，峰值 **11,865 MiB**；4B **6.106 s / 3.981 s**，峰值 **16,480 MiB**；mask shape 恒等于输入尺寸 `1080×1920` | 本会话实测 |
| spark2 `freeform`（Grounded Caption/VQA）逐字响应 | 2B **2.164 s** / 4B **3.981 s**（seabreeze 1.75 MP） | 2B：`<ref>A blond-haired man</ref><box><9><69><756><1000></box> wearing <ref>a striped shirt</ref><box>…</box> leans against <ref>a wooden railing</ref><box>…</box> overlooking the ocean .`；4B：`<ref>A man</ref>… with <ref>blond-hair</ref><box><452><62><756><481></box> and <ref>headphones</ref><box>…</box> sits on <ref>a bench</ref>…` → **prose 里内联 ref/box**，同一 label-block 正则可解析（每 ref 一框，无 `<OBJ>` → mask 不触发），4B 区域粒度更细 | 本会话实测 |
| spark2 `matting` query 词表敏感性（seabreeze） | 2B / 4B | `sailboat` `sail` `sailing boat` → **2B 与 4B 全部 `none`**（2B 0.58–0.61 s，4B 0.93–0.98 s）；`boat` → 双双命中出 alpha；`yacht` → 2B `none`，**4B 命中**（1.882 s，mean 0.0749）→ 同一目标的词表命中随模型尺寸变化 | 本会话实测 |
| spark2 mask 与 box 的关系（`seabreeze.jpg` alpha 逐像素 [M]） | 2B/4B | mask 质量几乎全在 box 内（box 内 mean **0.39**，box 外 mean **0.0–0.0009**），但 **box 外最大值达 1.0（2B `boat`：护栏 box 之外的桅杆）**，4B box 外最大 0.27–0.62 → **alpha 覆盖对象本体，box 只是定位框** → 服务端**不得把 mask 裁到 box** | 本会话实测 |
| spark2 **稳态压测**（warmup 1 次 + 每任务 6 次重复，`p50/p95/min/max` + 逐次 `max_memory_allocated` [M]） | 2B bf16 | box 族 p50：**point 0.56 / referring 0.77 / detection(2 label, 4 实例) 0.97 / ocr 3.93 / count_box(16 box) 3.00 / count_point(chairs 密集) 0.50–0.96**；像素族 p50：**segmentation@0.71MP 1.46（峰值 11,291 MiB）/ matting@1.75MP 1.24（7,511 MiB）/ segmentation@2MP 2.13–2.68（11,866 MiB）**；warmup 全部 1.2–3.9 s（比稳态慢 20–50%），稳态方差不超过 ±0.1 s（`count_point` 与 2 MP segmentation 有 2–3× 的尾部抖动，max 分别 0.958 / 2.836） | 本会话实测 |
| spark2 **稳态压测**（同上） | 4B bf16 | box 族 p50：**point 1.25–2.04 / referring 1.44 / detection 1.84 / count_box 8.3–11.1 / count_point 10.6–12.8 / ocr 15.3**；像素族 p50：**segmentation@0.71MP 4.76（峰值 20,540 MiB）/ matting@1.75MP 3.97（16,765 MiB）/ segmentation@2MP 6.54（21,124 MiB）**；引擎 load 67.3 s，load 后驻留 13,941 MiB。**4B 全能力峰值 21.1 GiB 高于 §5.1 单请求的 16.5 GiB**（本压测进程内 pixel decoder 权重与两引擎权重共存，见下条）→ 4B 生产实例的显存预算应按 **21 GiB** 留 | 本会话实测 |
| spark2 12 请求**顺序队列**（box+像素混合，单 engine 常驻，无排队调度） | 2B / 4B | 2B：**15.36 s 完成 12 请求 = 0.78 rps**，队列内 p50 0.853 / p95 3.435；4B：**55.32 s = 0.217 rps**，队列内 p50 3.09 / p95 13.063 → 单进程吞吐 ≈ **1 请求/秒（2B）**，API 容量规划用这个数而不是单请求延迟 | 本会话实测 |
| spark2 `predict_batch` 8 图批 vs 逐张（`detection dogs`，同一进程） | 2B / 4B | 2B：批 **1.92 s（0.24 s/图）** vs 逐张 **6.19 s（0.774 s/图）** → **批处理 3.2× 吞吐，峰值显存几乎不变（4,867 MiB）**；4B：批 **6.15 s（0.769 s/图）** vs 逐张 **23.75 s（2.969 s/图）** → **3.8×** | 本会话实测 |
| spark2 **2B + 4B 双引擎常驻**（同进程，先后各跑一次 segmentation@0.71MP） | 合并分配器峰值 | **20,540 MiB**（≈20 GiB），两引擎共存 + 一次 4-mask 解码；2B 单独跑时 20,539 MiB（4B 权重已在内存）→ 主机上同时挂 2B 与 4B 全能力实例需 **≥24 GiB 显存预算** | 本会话实测 |

### 5.2 4070-class 服务器推算（RTX 4070 12 GB，501.7 GB/s 显存带宽）[I] —— 本机禁止跑模型（`AGENTS.md` 红线），本会话没有 4070 主机可测；稳态实测在 §5.1 的 spark2 压测行

decode 是权重带宽瓶颈（每 token 读完全部权重）：

| 权重 | 字节 | ms/token | 理论 tok/s |
|---|---|---|---|
| 4B bf16 | 9.66 GiB | 19.3 | ~52 |
| 2B bf16 | ~4.0 GiB | 8.0 | ~125 |
| 4B Q4_K_M | ~2.4 GiB | 4.8 | ~208 |

Prefill（2 × 参数 × token 数，稠密 bf16 算力 ~14.3 TFLOPS）：1,100 token ≈ 10.7 TFLOP → **0.75 s**。

组合估算：
- box-only 小响应（本会话实测 2 实例 grounding = 23 token）：**视觉编码 0.1–0.3 s + prefill 0.05–0.75 s + decode 0.44 s ≈ 0.5–1 s**
- 像素任务（segmentation/matting）：**GB10 实测 2B 2.51 s / 4B 8.57 s @0.71 MP（4 对象）**[M]；4070 有独立显存与更高 decode 带宽 → 估算 **2B ≈ 2–3 s、4B 全能力 ≈ 2–4 s** [I]。显存：2B 全能力峰值 **11.3 GiB [M]**（12 GB 卡刚好放得下），4B 全能力峰值 **15.9 GiB [M]**（12 GB 卡不可行，16 GB+ 必需）。**4B 像素任务没有量化逃生路径**（Q4 丢 mask，官方确认）→ 12 GB 卡的像素能力只能靠 2B

VRAM 实测（GB10，`torch.cuda.max_memory_allocated` [M]，2B → 4B 同顺序）：freeform 无 mask @1.75 MP **5,059 → 9,780 MiB**；matting 1 mask @1.75 MP **7,515 → 12,126 MiB**；segmentation 4 mask @2 MP **11,865 → 16,480 MiB**。权重（2B ≈4 GiB，4B 9.66 GiB）之外的 S 网格激活随像素数与对象数增长（4B @2 MP 激活 ≈6.7 GiB）。**12 GB 卡边界：2B 全能力 @2 MP 用 11.6 GiB 刚好放得下；4B 全能力 16 GiB+ 不可行**，16 GB 卡的 4B 像素任务应把 `max_pixels` 压到 ≈1.7 MP（实测峰值 12.1 GiB 有余量）[I]。
- **与 2B 稳态实测对照（spark2 [M]，§5.1 压测行）**：2B box 族 p50 实测 `point` 0.56 / `referring` 0.77 / 2-label `detection` 0.97 / `count_box`(16 box) 3.00 / `ocr` 3.93，与上面 4070 的 0.5–1 s 小响应估算吻合 → **4070-class 主机跑 2B box 族不需要额外余量**；4B 侧上面 2–4 s 的像素估算低于 GB10 稳态实测（3.97–6.54 s），容量规划应直接采用 GB10 数字（保守）

---

## 6. 部署矩阵

| 路径 | box/point/OCR/文本 | mask/alpha | 并发 | 约束 |
|---|---|---|---|---|
| `transformers` + `WeVisOneVL` | ✅ 全 11 任务 | ✅ | `predict_batch`（VLM 批生成，mask 逐图解码）；单进程常驻 | `attention_implementation` 默认 sdpa（flash-attn 更快，Linux 首选）；bf16 CUDA |
| `vLLM`（`scripts/serve_vllm.py`，OpenAI 兼容） | ✅ | ❌（pixel decoder 需要 hidden-state hook，vLLM 路径不支持） | vLLM 连续 batching；`limit_mm_per_prompt` 10 images | 仅 Linux；`max_model_len` 默认 8,192；`gpu_memory_utilization` 0.45 |
| Ollama / llama.cpp GGUF | ✅ | ❌（官方确认 masks 丢失；GGUF 侧 feature/hidden-state decoder 接口未验证 [3P]） | llama.cpp server 单/多 worker | 4B Q4_K_M 4.38 GiB 峰值实测 → 8 GB 卡可跑 box-only |
| LiteRT-LM 2B（官方转换） | ✅ | ✅（转换含 pixel decoder） | 端侧单请求 | Android/Mac；TTFT 与 tok/s 见 §5.1 |

---

## 7. API 设计建议

### 7.1 端点

沿用官方 `TASKS` 命名，不自造词表：

```
GET  /v1/tasks            能力自描述：11 个 task key + prompt 模板 + 输入/输出 schema + 示例
POST /v1/perceive         统一入口
POST /v1/health           就绪探针（engine 已加载 / VRAM 余量 / 模型 id / 能力集）
```

`POST /v1/perceive` 请求：

```json
{
  "image": {"url": "...", "b64": "...", "path": "..."},
  "task": "detection",                      // 11 个 canonical key 之一
  "query": "dogs\npeople",                  // 字符串或标签数组
  "reference_boxes": [[12, 34, 90, 120]],   // 仅 visual_prompt，像素 xyxy
  "options": {
    "max_new_tokens": 1024, "temperature": 0.0,
    "decode_matting": true, "min_pixels": 16384, "max_pixels": 2073600,
    "return_overlay": true, "mask_format": "png"
  },
  "idempotency_key": "sha256:…"
}
```

响应：

```json
{
  "request_id": "…", "task": "detection", "mode": "box",
  "image_size": [1920, 1080], "size_after_preprocess": [1904, 1064],
  "visual_tokens": 1974,
  "groups": [
    {"label": "boys",
     "absent": false,                     // 逐 label 拒答（<ref>people</ref><box>none</box>）
     "instances": [
       {"box_norm": [0.012, 0.340, 0.090, 0.620],   // 0–1，模型原生
        "box_px":   [23, 367, 173, 670],            // 服务端换算（对原图尺寸）
        "point_px": null,
        "mask": {"url": ".../m0.png", "format": "png", "mode": "alpha"},
        "decode_ok": true}
     ]}
  ],
  "declared_total": 6,      // count_* 的 "Total number: N." 行；其余任务 null
  "count_by_items": 6,      // 解析出的实例总数（逐 label 求和）
  "total_consistent": true, // false 时 agent 应优先 count_by_items（4B 实测会数错）
  "warnings": ["label 'people' absent", "malformed closing tag: label 'chairs'"],
  "raw_response": "<ref>boys</ref><box><12><340><90><620>,…</box>\nTotal number: 6.",
  "latency_ms": {"model": 620, "total": 910}
}
```

### 7.2 服务端必做清单

1. **Label 块解析器**（照抄 `demo/app.py:614` `parse_predictions_with_masks` [M]）：`<ref>(.*?)</ref>\s*<(box|point)>(.*?)</\2>`；body 按 `,` 切段、段内 `<(\d{1,4})>` 取数、`step = 2 if point else 4`；`none` → 该 label `absent: true`；**闭合标签需容错配对**（4B 实测输出 `<point>none</box>` 畸形闭合，backreference 匹配不到 → label 丢失）
2. **计数一致性校验**：解析 `\nTotal number: N.` 得 `declared_total`，与 `count_by_items` 比对；不一致置 `total_consistent: false`（4B 实测 `chess.jpg count_point` 声明 15/2B 16，密集场景 4B `count_point` 吐 31 points）
3. **归一化→像素换算 + 预处理尺寸记录**（resize 后尺寸必须回传，否则 agent 坐标不可对齐）
4. `ImageOps.exif_transpose` + RGBA 合成（服务端做，写进文档）
5. mask float32 → 8-bit PNG（alpha 任务输出真 alpha PNG；segmentation 输出二值 mask PNG），逐对象 URL/base64
6. 常驻 warm engine（`WeVisOneVL(model_name_or_path=…)` 一次加载，`mask_decoder.pt` 随 repo 下载）；box-only 与像素任务**分队列**（像素任务 ≈3× 耗时 + 显存峰值）
7. `latency_ms.model` 用 `InferenceResult.latency_s`（模型侧，不含排队）[M]，`total` 用服务端墙钟
8. 缓存 key = `sha256(image bytes) + task + canonical prompt`（prompt 由 `build_prompt` 生成，天然规范化）
9. `GET /v1/tasks` 返回每个 task 的 canonical prompt 模板与一个真实示例（agent 可发现能力，也是 prompt 纪律的唯一来源）
10. OpenAPI schema 自动生成；拒绝未知 `task`（engine 的 `ValueError` 消息就是权威白名单）
11. 显式处理 `decode_masks` 的 `RuntimeError`（teacher-forced 响应无 `<OBJ>`）→ 返回 `partial` 状态而非 500
12. **`POST /v1/matting_box`**（`image` + `box` 归一化 xyxy）→ 服务端合成单 `<OBJ>` 响应，调 `decode_masks(image, build_prompt("segmentation", phrase), synth_response, boxes=[client_box])`（§4 陷阱 8 的机制，spark2 实跑数据在 §5.1）。README 案例研究的 Box-prompted Matting 在 released prompt 表里没有模板，这是唯一可行封装路径。
13. **mask 后处理纪律**（逐像素实测 [M]）：不裁到 box（对象可超出 box，2B `boat` 在 box 外有 full-alpha 像素）；不二值化 alpha；每对象输出 `mask_box_mass`（box 外 alpha 质量占比），作为 agent 可读的 `leakage` 信号

### 7.3 规模与配额

- 匿名调 HF demo 实测约 **7 次请求** 后 ZeroGPU 配额耗尽（2B 与 4B 空间共享同一配额池）[M] → 原型验证可以用 demo，生产必须自托管
- 显存预算（spark2 稳态压测，逐次 `max_memory_allocated` [M]）：**2B 全能力实例峰值 11.9 GiB**（2 MP，4 mask）、1.75 MP 单 mask **7.5 GiB**、box-only 仅 **4.9 GiB**；**4B 全能力实例 21.1 GiB**（2 MP 峰值，含 pixel decoder 常驻；单请求首测口径 16.5 GiB，压测口径 21.1 GiB 是生产预算）、1.75 MP 单 mask **16.8 GiB**、box-only **14.2 GiB**；**2B+4B 双引擎常驻 = 20.5 GiB** → 12 GB 卡只能跑 2B（全能力须把 `max_pixels` 压到 ≤1.75 MP，7.5 GiB 有余量），16 GB 卡跑 2B 全能力 @2 MP 刚好，4B 全能力需 24 GiB 级
- 吞吐实测（spark2 单进程常驻，12 请求 box+像素混合顺序队列 [M]）：2B **15.36 s / 12 = 0.78 rps**（队列内 p50 0.85 s、p95 3.44 s），4B **55.32 s / 12 = 0.217 rps**（p50 3.09 / p95 13.06）→ **容量规划用 rps 而不是单请求延迟**。`predict_batch` 8 图批实测（`detection dogs`，同进程）：2B **1.92 s（0.24 s/图）vs 逐张 6.19 s（0.774 s/图）= 3.2×**，4B **6.15 s（0.769 s/图）vs 23.75 s（2.969 s/图）= 3.8×**，显存增量几乎为零 → box-only 端点应该走批处理
- 冷启动必须算进 SLA（spark2 实测 [M]）：**引擎 load 2B 34.4 s / 4B 67.3 s**，load 后驻留 4.7 GiB / 13.9 GiB；warmup 首请求比稳态慢 20–50%（实测 warmup 1.2–3.9 s vs 稳态 p50 0.56–3.93 s）→ 服务进程常驻 + 预热完毕再放流量，滚动重启要按 load 时间排窗口
- **服务端最小依赖（逐字 import 图 [M]）**：`torch`、`numpy`、`Pillow`、`transformers==5.12.1`（`engine.py:124` 用 `AutoModelForImageTextToText`/`AutoProcessor`）、`huggingface-hub==1.21.0`（`engine.py:85` 的 sidecar 下载）、`qwen_vl_utils==0.0.14`（`engine.py:241/327` 的 `process_vision_info`，即 smart_resize 预处理）。`matplotlib`/`supervision`/`scipy`/`pycocotools` 只在 demo 与评测路径（`visualization.py:11` 用 `supervision`）→ **生产镜像可裁掉这四项**
- **Python 版本是硬约束（本会话实测 [M]）**：pinned `matplotlib==3.11.0` 要求 `Requires-Python >= 3.11`，在 Python 3.10 上 `pip install -r requirements.txt` 直接 `No matching distribution found for matplotlib==3.11.0` 失败（尽管 pyproject 写 `requires-python >= 3.10`，自相矛盾）。裁掉 demo 依赖后 py3.10 可跑引擎；按 requirements 安装则必须 **Python >= 3.11**（本会话用 3.13 venv）

---

## 8. 已知局限（写进 API 文档，避免 agent 踩）

- 无 CoT、无视频、引擎内无多图、无 polygon 输出；mask 是逐对象概率图
- `count_text` 未实现
- **`matting`/`referring` 的 query 词表敏感（实测）**：`seabreeze.jpg` 上 `sailboat`/`sail`/`sailing boat` → 2B 与 4B 全部 `none`；`boat` → 双双命中出 alpha；`yacht` → 2B `none`、4B 命中。API 语义：`none` 是合法响应（不是错误），agent 应换词重试；服务端不得把 `none` 变 500
- prompt 必须走固定模板族；`referring` query 是单短语，`detection` 是标签列表，`visual_prompt` 的参考框是像素坐标
- 无置信度 → agent 侧需用 box 面积/数量做后过滤
- 量化坐标 1/1000 的固有分辨率误差
- 2 MP 像素预算上限：更高分辨率输入会被 resize，小目标召回下降
- GGUF 量化精度：PALLER-NOVUS 的 272-query 全扫描还没完成 [3P]。本会话在 spark2 上补测了 COCO 60 图配对：4B Q4_K_M 比 4B BF16 低 F1 mean 2.1–5.3、拒答率低 1.6–3.3 个点（§12.1 [M]）；LiteRT 路径未测
- 2B 拒答率 92.9% 明显低于 4B 96.4%；2B 在 PointBench 62.9 vs 4B 71.9 → **选 2B 省钱时，指向/拒答类任务精度显著下降**
- alpha 输出是**连续软边**（`float32` 概率图；8-bit PNG 全 256 级；(0.05, 0.95) 像素占比 2.87%（2B person）/ 3.23%（4B person）[M]）→ **服务端绝不在服务端二值化 alpha**，segmentation（二值语义）与 matting（alpha 语义）共用同一 decoder，输出编码必须按 task 区分
- `freeform` 的响应是 **prose 内联 `<ref>…</ref><box>…</box>** [M]，不是纯文本 → 解析器必须支持内联单框 ref；4B 区域粒度显著更细（seabreeze：4B 拆出 `blond-hair`/`headphones`/`bench`，2B 只 3 个区域）
- 密集场景 4B 实测退化：`count_point` 过度计数（livingroom `chairs` → 31 points，2B 为 1）、`count_box` 数字错（声明 0 而实际 2）+ 畸形闭合标签 → **数字/坐标一致性校验不可省**
- **mask 覆盖对象本体、可超出 box**（实测：box 内 mean 0.39 / box 外 mean 0.0–0.0009，2B `boat` box 外最大 1.0，即护栏外桅杆像素）→ 裁剪 mask 到 box 会切掉目标；box 外质量是**目标延伸**而非泄漏，API 文档必须写明这一点

## 9. 建议落地顺序

1. **安装只发生在远程 GPU 主机，本机禁止跑模型**（见 `AGENTS.md` 红线：本会话在本机装 CUDA torch + 权重并跑推理，把机器搞死机，所有本地安装已清除）。`wevisone-vl` **不在 PyPI**（本会话实测：`pip install wevisone-vl` → `No matching distribution found` [M]）。安装方式是 repo 逐字指令（README:201 [M]）：`git clone` → `pip install -r requirements.txt && pip install -v -e .`（pyproject 名 `wevisone-vl` 1.0.0，import 名 `wevisone_vl`）。依赖**全部精确 pin**：`transformers==5.12.1`、`torch==2.11.0`、`torchvision==0.26.0`、`huggingface-hub==1.21.0`、`accelerate==1.14.0`、`qwen_vl_utils==0.0.14`、`pycocotools==2.0.11`、`supervision==0.31.0`、`scipy==1.18.1`、`matplotlib==3.11.0`、`Pillow==12.2.0`、`safetensors==0.8.0`；`vllm`/`triton` 带 `sys_platform != 'win32'` 标记。两个坑（本会话实测 [M]）：**pinned `matplotlib==3.11.0` 要求 Python >= 3.11**（py3.10 装 requirements 直接 `No matching distribution found`）；**PyPI Windows 的 `torch==2.11.0` 是 CPU 轮子**（`2.11.0+cpu`，`cuda False`）—— 服务器需 `--index-url https://download.pytorch.org/whl/cu126` 才有 CUDA 版。权重在服务器上拉：`git clone --depth 1 https://huggingface.co/tencent/WeVisOne-VL-2B`（git-lfs 直接落 `model.safetensors` 4.88 GB + `mask_decoder.pt` 38 MB，本会话实测 [M]，无需 huggingface-cli）
2. 写全量 box 解析器（§7.2 条 1–2 的语法与校验）+ FastAPI `/v1/perceive`，用 `demo/examples/images/*.jpg`（dogs/boys/chess/coffee_sign/livingroom/parasol/seabreeze）做 11 任务冒烟 —— 逐字响应基线已在本会话于 spark2 取得（`probe2.py` / `probe4.py` / `probe5.py`，§5.1 的 raw response 就是现成对拍基准）
3. 与 HF demo 输出逐字对拍（`raw_model_output` 端点可取 [M]）验证 prompt/预处理一致性
4. **压测主机 = spark2**（`ssh spark2` = `acro@10.66.0.12` 免密，DGX Spark GB10 128 GB 统一内存，工作目录 `~/Documents/vlocate`）。**已完成**：`bench/bench.py` 稳态压测（warmup 1 + 6 reps × 9 任务 × 2 模型，12 请求顺序队列，`predict_batch` 8 图批对照，2B+4B 双引擎常驻）→ §5.1 压测行、§5.2 对照、§7.3 吞吐/显存/冷启动全部 [M]。本机不跑模型、不做显存测量（`AGENTS.md` 红线）
5. 加缓存、队列分层、`/v1/tasks` 自描述，接 agent

## 10. 参考

- 仓库：<https://github.com/Tencent/WeVisOne-VL>（Apache-2.0，代码+权重+Cookbook+评测管线；TODO 仅剩 fine-tuning 代码）
- 项目页：<https://tencent.github.io/WeVisOne-VL>；论文 arXiv:2610.31329
- 权重：<https://huggingface.co/tencent/WeVisOne-VL-2B> / `tencent/WeVisOne-VL-4B`（apache-2.0，`Qwen3VLForConditionalGeneration`）
- Demo：<https://huggingface.co/spaces/hugging-apps/wevisone-vl-2b> / `wevisone-vl-4b`（zero-A10G）
- `mask2alpha` 分支（matting 标注器，mask→alpha 数据引擎）：<https://github.com/Tencent/WeVisOne-VL/tree/mask2alpha>
- 量化与端侧：`PALLER-NOVUS/WeVisOne-VL-GGUF`（BENCHMARKS.md，20-query pilot）、`litert-community/WeVisOne-VL-2B`（含 pixel decoder 的官方转换，实测数字出处）
- 模型卡使用边界原文：“not intended to make autonomous high-impact decisions or to identify individuals”

## 11. 验证状态

**本会话完成的验证**：engine.py/prompts.py 逐字读码（11 任务表、模板族、`parse_seg_boxes`、sidecar 解析、pixel decoder 流程、`decode_masks` 的 `boxes` override 语义、默认参数）；4B config.json；2B/4B demo 的 `gradio_client.view_api` 端点签名与 task label 映射；4B demo 实测 7 请求（服务端 1.5 s，端到端 2.7–8.1 s，匿名 IP 约 7 次后 ZeroGPU 配额耗尽）；本机硬件探测（RTX 4070 12 GB / 48 GB RAM）；**spark2（DGX Spark GB10，venv `transformers 5.12.1 + torch 2.11.0 nvidia`，2B/4B bf16，`mask_decoder.pt` 随权重就位）**：(a) 26 请求逐字 raw response（box 族 18、像素族 8）+ 单请求 `latency_s` + `max_memory_allocated` 峰值 + alpha 软边统计；(b) **稳态压测 74 次请求**（9 任务 × 6 重复 × 2 模型 + 12 请求顺序队列 + `predict_batch` 8 图批对照 + 2B/4B 双引擎常驻），逐任务 p50/p95/min/max/warmup/逐次显存峰值，逐字结果在 `results/bench_results.jsonl` → §5.1 压测 5 行、§7.3 吞吐/显存/冷启动三条全部由 [I] 转 [M]

**本轮新增验证（2026-10-30，spark2）**：4B Q4_K_M 在打补丁的 llama.cpp 上实跑（`bed0a85` + `runtime.patch`，CUDA 13.0，`sm_121`），与 2B/4B BF16 原生引擎在 COCO 60 图上配对，测了两种像素预算：6 个臂 × 120 请求 = 720 次（`results/arms.tgz`）；2B 的输入尺寸、批大小、label 数、query 长度扫描（`results/limits_2B.jsonl`）；`docker/` 镜像在 spark2 上构建并用 GPU 冒烟。

**未验证**：4070-class 硬件的真实 p50/p95 与显存峰值（§5.2 仍是 [I]：本会话唯一可用的测试主机是 spark2 GB10，没有 4070；本机按 `AGENTS.md` 红线禁止跑模型）；2B Q4 与 Q5/Q8 等其他量化；更难的拒答集（WeRef-Bench）上 Q4 的表现；amd64 镜像在 GPU 上的运行冒烟（CI 只验证构建与推送）；`mask2alpha` 分支实跑（与 API 无关）；多 engine 进程的并发扩展。后续所有模型侧测量都只在 spark2 的 `~/Documents/vlocate` 进行，产物取回本机。

---

## 12. 部署决策补充（2026-10-30，spark2 实测）

### 12.1 选型：4B Q4_K_M vs 2B BF16 → **选 2B BF16**

配对实验 [M]：COCO val2017 子集 60 张图、355 个 GT box（种子 20261030，非 crowd，每图 2–20 个实例、≤4 个类别）。每张图跑两个请求：`detection`（标签 = 图中出现的类别）和 `referring`（随机选一个图中不存在的类别，测拒答）。box F1 用 IoU 0.5 与 0.5:0.95 均值，口径与官方相同。Q4 用打补丁的 llama-server（`-ngl 99 -c 8192 --parallel 1`），BF16 用官方 `WeVisOneVL.generate_text`。两种像素预算都测：

| 臂 | 视觉 token 预算 | F1@0.5 | F1 mean | 拒答率 | detection p50 / p95 | 驻留显存 |
|---|---|---|---|---|---|---|
| 2B BF16（原生） | 16–2025（引擎默认） | 76.1 | 55.6 | 100% | 1.09 / 2.44 s | 4.73 GiB（box-only 峰值，`max_memory_allocated`） |
| 4B BF16（原生） | 16–2025 | 76.6 | 57.9 | 100% | 1.96 / 4.32 s | 14.2 GiB |
| 4B Q4_K_M（llama.cpp） | 16–2025 | 74.2 | 52.6 | 96.7% | **0.69** / 1.35 s | 4.64 GiB（进程，`nvidia-smi`） |
| 2B BF16（原生） | 1024–1960（官方评测 / 模型卡口径） | 79.6 | 59.9 | 98.3% | 1.31 / 2.90 s | 同上 |
| 4B BF16（原生） | 1024–1960 | **82.0** | **64.3** | 98.3% | 2.23 / 5.53 s | 同上 |
| 4B Q4_K_M（llama.cpp） | 1024–1960 | 79.7 | 62.2 | 96.7% | 1.16 / 2.10 s | 4.81 GiB |

结论：
- **box 精度：Q4 与 2B 打平，没有决定性优势**。预算一致时，F1@0.5 两者分别是 79.7 / 79.6（1024 档）和 74.2 / 76.1（默认档），F1 mean 在 1024 档 Q4 高 2.3、默认档低 3.0。量化让 4B 掉了 2.1–5.3 F1 mean，把 4B 相对 2B 的优势基本抵消了。60 图的抽样误差约 ±3 F1 [I]，这个量级上两者无法区分
- **拒答：Q4 更差**。BF16 在两种预算下拒答率都是 98.3–100%，Q4 两种预算都是 96.7%，都出现了把 `cow`、`frisbee`、`parking meter` 当成存在、并给出一个大框的情况。随机挑的 COCO 缺失类别是容易的负样本，官方 WeRef 这类难负样本上差距只会更大 [I]
- **显存：没有收益**。Q4 4B 的 llama-server 进程占 4.64–4.81 GiB，2B BF16 做 box 任务的峰值是 4.73 GiB（0.71 MP；2 MP 时 5.01 GiB），几乎一样。PALLER 那份 RTX 2060 上 4.38 GiB 的数字也是同一量级
- **能力：Q4 少两项**。segmentation 和 matting 在 GGUF 路径上不可用（官方与 PALLER 都已确认）；2B BF16 支持全部 11 个任务
- **延迟：Q4 快，但主要是后端差异**。同一预算下 Q4 比 2B 快 11–37%（1.16 vs 1.31 s，0.69 vs 1.09 s），这里有一大部分来自 llama.cpp 与 HF `generate` 的实现差别，并不只是量化的效果。2B 用 `predict_batch` 批处理后每张 0.20–0.23 s（§12.2），吞吐反而超过单槽的 llama-server
- **运维：Q4 需要第二套栈**。要额外维护打过补丁、钉死 commit 的 llama.cpp；PALLER 自己也写明“unpatched upstream untested”；Q4 输出里会带 `<|im_end|>`（需要 `--special`），解析器还得多一条剥离规则
- **只有一种情况选 Q4**：显存 ≤ 6 GB 的边缘设备，而且只要 box/point。本 API 要提供完整的 11 项能力，所以**定 2B BF16**。如果精度优先、显存 ≥ 24 GiB，选 4B BF16（F1 mean 比 2B 高 4.4，延迟约 1.7 倍）
- **附带发现：视觉 token 下限对精度影响很大**。COCO 图约 0.3 MP，引擎默认 `min_pixels = 16×1024` 时不会放大，只有约 300 个视觉 token。把下限提到 1024 token（`min_pixels = 1024×1024`）后，2B F1 mean +4.3、4B +6.4，代价是延迟约 +20%（2B 1.09 → 1.31 s）。**生产配置建议用 `min_pixels = 1,048,576`、`max_pixels = 2,007,040`（1960 token，官方评测口径）**。像素任务在这个下限下的表现还没测

### 12.2 同步 + 异步批量：**建议加，作为第二阶段端点**

需要异步的原因（实测）：
- agent 的工具调用一般有 30–120 s 的超时。2B 跑 100 张图的 box 任务约需 100 × 0.23 = 23 s（批处理），像素任务约 100 × 0.84 = 84 s。4B 再乘 3–4。同步连接撑不住
- 合批才能拿到吞吐：box 任务单张 0.77 s，16 张一批时每张 0.23 s（3.4×），32 张一批时 0.20 s（3.9×）。同步请求来得零散，凑不成批；异步队列可以把同一 `(task, query)` 的条目攒成一批

端点设计（形态参考 OpenAI Batch API：JSONL、`custom_id`、24 h 完成窗口、可取消 <https://developers.openai.com/api/docs/guides/batch>）：

| 端点 | 语义 |
|---|---|
| `POST /v1/perceive` | 同步，单图，服务端超时 60 s。保持已审核通过的契约不变 |
| `POST /v1/batches` | body 为 `{items: [{custom_id, image: {url｜base64｜file_id}, task, query, options}], webhook_url?, metadata?}`，或者一个 JSONL `file_id`。返回 `202 {batch_id, status: "queued", item_count}`。支持 `Idempotency-Key` 头 |
| `GET /v1/batches/{id}` | `status ∈ queued｜running｜completed｜failed｜cancelling｜cancelled｜expired`，以及 `counts {total, succeeded, failed}`、`created_at/started_at/completed_at` |
| `GET /v1/batches/{id}/results` | 流式 JSONL，每行 `{custom_id, status, result｜error}`。`result` 的结构与 `/v1/perceive` 的响应逐字段相同；mask 以 `mask_url`（PNG）给出，不内联 |
| `DELETE /v1/batches/{id}` | 取消：已完成的条目保留，未开始的标为 `cancelled` |
| webhook | 完成时 POST `{batch_id, status, counts}`，带 HMAC 签名头；agent 也可以只轮询 |

调度规则：
- **两条队列，同步优先**。单 GPU 单 engine 时，同步请求优先级最高，批量条目在空档里以 GPU 微批执行，这样交互式 agent 的 p95 不受批量任务拖累。2B 混合负载实测 0.78 rps（§7.3），批量队列的预计完成时间按条目数除以每张耗时估算，并通过 `GET` 返回
- **微批上限（实测 [M]）**：box 任务 n = 16。32 时每张只比 16 再快 13%，峰值显存从 7.0 涨到 9.4 GiB，而且出现了**同一张图在 32 批内输出不一致**（`all_same: false`，bf16 padding 带来的数值差）。像素任务 n = 8：每张 0.84 s 最优，16 时反而 0.94 s；`predict_batch` 的 mask 是逐图解码的，峰值恒定 11.0 GiB
- **合批键是 `(task, query)`**。`predict_batch` 只接受一个 `task/query`，不同 query 要走 `prompt=[...]` 的逐条 prompt。同一批内图片分辨率差异大时 padding 浪费会增加，服务端应先按视觉 token 数分桶
- **API 文档必须写明**：批量结果和同步结果在坐标 token 上可能差 ±1–2（批内数值差），两者不保证逐字一致

配额与上限：

| 项 | 上限 | 依据 |
|---|---|---|
| 每批条目 | **500** | 2B box 500 × 0.23 s ≈ 2 min；像素 500 × 0.84 s ≈ 7 min；4B ×3–4 后仍能在 30 min 内完成 |
| 每个 key 同时运行的批 | 2 | 单 GPU 只有一个 engine，再多也只是排队 |
| 请求体 | 200 MB（base64 内联），建议用 `url` / `file_id` | 与 OpenAI Batch 的输入文件上限同级 |
| 完成窗口 / 结果保留 | 24 h / 7 d | 过期的条目标 `expired`，不静默丢弃 |
| 同步端点图片数 | 1 | 多图走批量，避免同步请求占住 GPU |

### 12.3 输入验证

图像（违反时返回 4xx，不能让引擎抛 500）：

| 检查 | 规则 | 依据 |
|---|---|---|
| 格式 | JPEG / PNG / WebP / BMP；GIF 和动图只取第一帧 | PALLER 实测 WebP 视频探测失败 → 统一先解码到第一帧 RGB |
| 字节数 | ≤ 20 MB（解码后的 base64 大小） | 2 MP 以上的像素不增加信息（§12.4） |
| 解码像素 | ≤ 50 MP，用 `Image.MAX_IMAGE_PIXELS = 50_000_000` 在解码前拦截，超限返回 413 | Pillow 默认值是 89.48 MP（实测），只告警不拒绝；48 MP 已实测可以处理 |
| 长宽比 | ≤ 200:1，超出返回 422 | `qwen_vl_utils.smart_resize` 中 `MAX_RATIO = 200`，超出直接 `raise`（读码 [M]） |
| 最小边 | ≥ 8 px | 小图会被 `min_pixels` 放大，但信息量不够 |
| 方向 | 先做 `ImageOps.exif_transpose` | 引擎只做 `convert("RGB")`，不处理 EXIF 方向，手机竖拍的图坐标会转 90°（读码 [M]） |
| 色彩 | RGBA / LA 合成到白底；CMYK / P / I;16 统一转 RGB | 引擎 `convert("RGB")` 会直接丢掉 alpha |

文本（label 数与长度都是实测的成本曲线 [M]，2B，`livingroom.jpg`，`detection`）：

| label 数 | 延迟 | 响应 token | 状态 |
|---|---|---|---|
| 1 | 0.34 s | 9 | 正常 |
| 5 | 1.17 s | 41 | 正常 |
| 10 | 2.24 s | 82 | 正常 |
| 20 | 4.48 s | 168 | 正常 |
| 40 | 14.0 s | 531 | **格式崩坏**（出现 44 个 `none`，label 重复） |
| 80 | **119.8 s** | **4096（打满上限）** | **截断，占住 GPU 2 分钟** |

每多一个 label 约增加 0.22 s，缺失的 label 也要输出一个 `<ref>x</ref><box>none</box>`。规则：
- `detection` / `count_*` / `point` 的标签：**≤ 20 个**，每个 ≤ 64 字符，去重并去掉首尾空白。需要更多标签时，由服务端拆成多个请求，或者让 agent 改用批量接口
- `referring` / `segmentation` / `matting` 的 query：**≤ 200 字符**。实测 200 字符的重复文本会让模型把 query 整段回显到 `<ref>` 里，导致响应 token 随 query 线性增长；4000 字符（888 prompt token）仍能输出，但没有意义
- `ocr_grounding` 文本 ≤ 200 字符；`freeform` prompt ≤ 2000 字符
- `visual_prompt` 参考框 1–8 个，必须是像素 xyxy、x1 < x2、y1 < y2，并且在图像范围内（越界先裁剪，面积为零则返回 422）
- **`max_new_tokens` 由服务端强制上限 1024**（引擎默认 4096 正是 80-label 那次跑 120 s 的原因）。触顶时响应带 `truncated: true`，不返回半截结果冒充完整结果
- 统一用 NFC 规范化，并剔除 `<`、`>`、`<|...|>` 这类控制序列，防止注入坐标 token 或 special token

### 12.4 大图：**服务端等比缩放到 `max_pixels`**

读码事实 [M]：VLM 的 processor 本来就会把输入压到 ≤ `max_pixels`（2,073,600），pixel decoder 的 CNN 分支固定把图 resize 到 **1024×1024**（`cnn_image_size = 1024`）。所以超过约 2 MP 的像素既不进 VLM，也不进 decoder，只在最后 `F.interpolate` 把 mask 放大回原图尺寸时用到。

实测（2B，`dogs.jpg` 用 LANCZOS 放大）[M]：

| 输入 | detection p50 | segmentation p50 | GPU 峰值（seg） | 4 个 mask 的 float32 体积 |
|---|---|---|---|---|
| 1024×696（0.71 MP） | 0.77 s | 1.55 s | 11,291 MiB | 11 MB |
| 1920×1080（2.07 MP） | 1.11 s | 2.85 s | 11,867 MiB | 33 MB |
| 3840×2160（8.3 MP） | 1.13 s | 4.47 s | 11,867 MiB | 133 MB |
| 6000×4000（24 MP） | 1.43 s | 3.28 s | 11,867 MiB | 384 MB |
| 8000×6000（48 MP） | 1.22 s | 3.18 s | 11,849 MiB | **768 MB** |
| 8000×6000 → 服务端先缩到 1662×1247 | — | **2.12 s**（另加 resize 0.22 s） | 11,849 MiB | 33 MB |

- **精度不受影响**：先缩小、再把 mask 双线性放大回 8000×6000，与直接输入大图相比，4 个 mask 的 IoU 为 **0.9993–0.9996**。box 坐标在 0.71–48 MP 的整个尺寸扫描里彼此相差 ≤ 2 个坐标 token（≤ 0.2%），但响应文本并不逐字相同
- **成本明显下降**：像素任务延迟 −33%，mask 体积 768 → 33 MB（23×）。GPU 峰值不变，因为它本来就被 `max_pixels` 封顶，这部分成本花在 CPU 内存、序列化和网络传输上
- 做法：解码并做 EXIF 校正后，如果 `w × h > max_pixels`，按 `s = sqrt(max_pixels / (w·h))` 用 LANCZOS 等比缩放；响应里返回 `input_size`、`processed_size`、`scale`
  - box / point 用原图尺寸换算（归一化坐标 × 原图宽高），对 agent 透明
  - mask 默认按 `processed_size` 编码成 8-bit PNG；需要原图分辨率时传 `mask_resolution: "original"`，由服务端放大，也可以让客户端自己放大
- 小图走反方向，即 §12.1 的 `min_pixels` 上调。两者合起来：输入统一落在 1024–1960 视觉 token，这也是官方评测的口径

### 12.5 容器化（`docker/`，已在 spark2 构建并冒烟 [M]）

文件：`docker/Dockerfile`、`docker/requirements.txt`（运行时最小依赖）、`docker/smoke.py`（加载 engine，跑一次 detection 和一次 segmentation）。

设计：
- 基础镜像 `python:3.12-slim-bookworm`。torch / torchvision 从 PyTorch 的 CUDA 索引安装（`ARG TORCH_INDEX`，默认 `cu130`），这些 wheel 自带 CUDA 与 cuDNN，**宿主机只需要 NVIDIA 驱动和 nvidia-container-toolkit**，不需要 CUDA 基础镜像
- `wevisone_vl` 以 `pip install --no-deps git+…@9a6bcd8` 方式钉死 commit 安装。demo / 评测依赖（gradio、matplotlib、supervision、scipy、pycocotools）不装，也就绕开了 `matplotlib==3.11.0` 对 Python ≥ 3.11 的约束
- **权重不打进镜像**：`-v /path/models:/models:ro`，再设 `WEVISONE_MODEL=/models/WeVisOne-VL-2B`。镜像不随模型变化，换 2B / 4B 只改环境变量。设了 `HF_HUB_OFFLINE=1`，运行时不会去访问 Hub
- 用非 root 用户（uid 10001）运行

spark2 实测结果：
- 构建 104 s，镜像 **5.3 GB**，arm64
- `docker run --gpus all` 识别到 `NVIDIA GB10`，`torch 2.11.0+cu130`，engine load 32.3 s
- detection 0.769 s，峰值 4,847 MiB；segmentation 1.533 s，峰值 11,291 MiB；4 个 mask 的均值为 `0.1326 / 0.0914 / 0.0864 / 0.0633`。**输出与 venv 内的逐字响应和 mask 统计完全一致**（venv 实测 0.1326、11,291 MiB）

生产运行要点：
- **每块 GPU 一个容器、一个 worker**（`uvicorn --workers 1`）。每个 worker 都会各自加载一份模型，多 worker 会让显存翻倍。横向扩展靠多容器加上游负载均衡
- 就绪探针在 load 完成并跑过一次 warmup 后才返回 200（load 32 s + warmup，§7.3），存活探针只检查进程。`docker compose` 里用 `deploy.resources.reservations.devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]` 申请 GPU
- **驱动版本**：cu130 的 wheel 需要驱动 ≥ 580（spark2 是 580.178.04 [M]）。驱动更老的主机用 `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128` 构建 [I，未测]
- **amd64 未构建** [I]：Dockerfile 里没有架构相关的步骤，torch 的 cu130 索引同时提供 x86_64 和 aarch64 wheel，但 x86 部署前需要在目标主机上重跑 `smoke.py`
- 批量结果和 mask PNG 写到挂载卷或对象存储，不写容器文件系统，保证容器重启后 `GET /v1/batches/{id}/results` 仍然可用

探针与产物：`bench/coco_prep.py`、`bench/cmp_arms.py`、`bench/score_arms.py`、`bench/run_arms.sh`（§12.1），`bench/limits.py`（§12.3–12.4），原始结果在 `results/arms.tgz`、`results/limits_2B.jsonl`。
