# 混合 SLM+LLM 边云协同推理解码（部署与运行指南）

本指南面向在本地单机环境复现论文《Hybrid SLM and LLM for Edge-Cloud Collaborative Inference》的核心算法：由 SLM 在边端生成草稿（draft），LLM 在云端对草稿进行一次性验证（verify），按设定阈值接受前缀，并在第一个不满足阈值的位置由 LLM 采样一个自由 token（free token）加入最终序列，形成迭代的 draft-verify 解码流程。

- 支持三种模式：`SLM only`、`LLM only`、`Hybrid SLM+LLM`。
- SLM 与 LLM 共享同一 `tokenizer`，保证编码/解码一致。
- 可配置 `block_size (k)`、接受阈值 `p_t`、`max_new_tokens`、`temperature`、`top_p`、模型名称、量化方式、设备等。
- 统计 cost：`cost = c_s * (N_s / T) + c_l * (N_l / T)`。

## 环境要求

- Python `3.10+`
- PyTorch `2.x`（可选 CUDA）
- Transformers（HuggingFace）
- 可选：`datasets`（加载 GSM8K），`bitsandbytes`（4bit/8bit 量化）
- 操作系统：Windows、Linux、macOS（本项目在 Windows 下可直接运行）

## 快速安装

- 进入项目目录：`d:\Scheme_1\token_level_coinference`
- 推荐使用虚拟环境（Windows）：
  - 创建：`python -m venv .venv`
  - 激活：`.venv\Scripts\activate`
- 安装依赖：`pip install -r requirements.txt`
- 验证安装（可选，运行 toy 单测）：`python -m unittest -q`

## 目录结构

```
.
├─ requirements.txt
├─ src/
│  ├─ config/
│  │  └─ default_config.yaml
│  ├─ models/
│  │  ├─ base_lm.py
│  │  ├─ slm_wrapper.py
│  │  └─ llm_wrapper.py
│  ├─ engine/
│  │  ├─ hybrid_inference.py
│  │  └─ cost_meter.py
│  ├─ data/
│  │  └─ gsm8k_loader.py
│  ├─ utils/
│  │  ├─ logging_utils.py
│  │  └─ text_utils.py
│  ├─ cli.py
│  ├─ run_baseline.py
│  └─ run_hybrid.py
└─ tests/
   └─ test_hybrid_engine.py
```

## 模型与 Tokenizer 配置

- 配置文件：`src/config/default_config.yaml`
- 关键字段：
  - `models.tokenizer_name`：共享的 tokenizer 名称（通常与 LLM 一致）
  - `models.slm_name`：SLM 模型名称（默认 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`）
  - `models.llm_name`：LLM 模型名称（默认 `openlm-research/open_llama_3b`）
  - `models.use_4bit` / `models.use_8bit`：量化加载（需 `bitsandbytes`）
  - `models.dtype`：`float16`/`bfloat16`/`float32`
  - `models.device`：`auto`/`cpu`/`cuda`
- 引擎参数：
  - `engine.block_size (k)`：SLM 草稿块长度
  - `engine.p_t`：LLM 接受阈值（越高越严格）
  - `engine.max_new_tokens`：最大生成长度
  - `engine.temperature`, `engine.top_p`：采样控制
- 成本系数：
  - `cost.c_s`, `cost.c_l`：与模型规模相关的权重，用于计算 cost

示例（保持默认）：
```
models:
  tokenizer_name: openlm-research/open_llama_3b
  slm_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  llm_name: openlm-research/open_llama_3b
  use_4bit: false
  use_8bit: false
  dtype: float16
  device: auto
engine:
  block_size: 4
  p_t: 0.2
  max_new_tokens: 64
  temperature: 0.7
  top_p: 0.9
cost:
  c_s: 1.0
  c_l: 10.0
runtime:
  seed: 42
```

## 运行模式与命令

- 统一入口（推荐）：
  - Hybrid：`python -m src.cli --mode hybrid --config src/config/default_config.yaml --prompts "Solve: 23+57"`
  - SLM 基线：`python -m src.cli --mode slm --config src/config/default_config.yaml --prompts "Explain gravity briefly."`
  - LLM 基线：`python -m src.cli --mode llm --config src/config/default_config.yaml --prompts "Explain gravity briefly."`

- 独立脚本：
  - Hybrid：`python -m src.run_hybrid --config src/config/default_config.yaml --prompts "What is 12 + 35?"`
  - SLM：`python -m src.run_baseline --mode slm --config src/config/default_config.yaml --prompts "What is 12 + 35?"`
  - LLM：`python -m src.run_baseline --mode llm --config src/config/default_config.yaml --prompts "What is 12 + 35?"`

- 加载 GSM8K 子集（可选，需 `datasets`）：
  - Hybrid：`python -m src.run_hybrid --use_gsm8k --num_samples 5`
  - Baseline：`python -m src.run_baseline --mode slm --use_gsm8k --num_samples 5`

## Hybrid 推理流程说明

- 算法位置：`src/engine/hybrid_inference.py:16` 的 `generate` 方法。
- 步骤概述：
  - SLM 按 `k` 生成草稿 token：`src/engine/hybrid_inference.py:23`
  - LLM 对 `ctx + draft` 前向一次，计算每个草稿 token 在 LLM 分布下的概率并按阈值接受：`src/engine/hybrid_inference.py:38-48`
  - 在第一个不满足阈值的位置用 LLM 采样一个 free token（分布索引为 `p + τ - 1`）：`src/engine/hybrid_inference.py:49-52`
  - 将已接受的 SLM token 与 LLM 的 free token 加入上下文，继续下一轮：`src/engine/hybrid_inference.py:52-58`

## 成本统计与日志

- 统计公式：`cost = c_s * (N_s / T) + c_l * (N_l / T)`
  - `N_s`：SLM 前向总次数（草稿生成每步）
  - `N_l`：LLM 前向总次数（每个草稿块一次）
  - `T`：输出 token 总数
- 运行日志示例（Hybrid）：
  - 文本输出：`Hybrid text: ...`
  - 计数：`Ns: <num> Nl: <num> T: <num>`
  - 平均成本：`Avg cost: <val>, Ns/T: <val>, Nl/T: <val>`

## 常见问题（Windows/本地）

- `bitsandbytes` 无法加载或无 GPU：
  - 将 `use_4bit`/`use_8bit` 设为 `false`，或安装支持的 CUDA 版本与显卡驱动。
- 显存不足（OOM）：
  - 减小 `dtype`（如 `float16`）、启用 8bit/4bit 量化、缩短 `max_new_tokens`、降低 `block_size`、选择更小的 LLM。
- 速度较慢（CPU）：
  - 使用更小的 LLM，或在有 GPU 的环境下设置 `device: cuda`。
- Tokenizer 不匹配：
  - 确保 SLM 与 LLM 使用同一 `tokenizer_name`，共享编码/解码。

## 验证与测试

- 运行 toy 单元测试：`python -m unittest -q`
- 测试文件：`tests/test_hybrid_engine.py`（验证 draft-verify 的接受与 free token 采样逻辑）。

## 下一步扩展

- 远程云侧前向：将 LLM 的 `forward_logprobs` 替换为远程 RPC 调用以模拟真实边云协同。
- 自适应阈值与块长：根据接受率动态调整 `p_t` 和 `k`，在吞吐与准确间取得平衡。