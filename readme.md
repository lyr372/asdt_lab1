# 项目简介（Scheme_1）

本仓库包含两类“协同推理”方案：
- 任务级协同推理（`task_level_coinference`）：以“本地小模型 Worker + 云端大模型 Supervisor”的协同为核心，负责任务拆解、检索与汇总。
- Token级协同推理（`token_level_coinference`）：以“SLM 草稿 + LLM 验证/修正”的解码算法为核心，提升吞吐与降低成本。

## 任务级协同推理（task_level_coinference）
- 双角色协同：入口 CLI 解析上下文并初始化本地/远程模型，然后启动对话与流式回调，见 `task_level_coinference/minions_cli.py:410`（`main`）与 `task_level_coinference/minions_cli.py:513-539`（初始化两端客户端）。
- 两种协议：
  - `Minion` 单轮-多步协同，云端给出提问与最终综合，见 `task_level_coinference/minions/minion.py:130`（协议入口 `__call__`）。
  - `Minions` 面向长文档的“代码即流程”式协同：Supervisor 产出代码块来“拆任务→生成 JobManifest→过滤/聚合结果”，本地批量执行，云端综合，见 `task_level_coinference/minions/minions.py:244`（协议入口）。
- 检索策略：支持 BM25 与语义嵌入检索；在 `Minions` 协议中通过参数开启并注入到代码块执行环境，见 `task_level_coinference/minions/minions.py:316-341`。
- 代码思路：
  - Supervisor 生成“准备任务”的代码块并执行以得到待处理的 `JobManifest`，见 `task_level_coinference/minions/minions.py:477-563`。
  - Worker（本地模型）并行处理每个任务/分块并返回结构化结果，见 `task_level_coinference/minions/minions.py:638-706`。
  - Supervisor 对结果进行 COT 思考与结构化综合，最终给出答案或请求补充信息，见 `task_level_coinference/minions/minions.py:836-939`。
  - 在 `Minion` 协议中，云端决策+本地执行的多轮交互封装在一个方法内，包含隐私屏蔽与 MCP 工具调用等能力，见 `task_level_coinference/minions/minion.py:405-476`（MCP 工具调用路径）。
- 示例应用：本地 RAG 文档检索（BM25/嵌入），结合 `Ollama` 运行本地模型，详见 `task_level_coinference/README.md`。

## Token级协同推理（token_level_coinference）
- 算法核心：SLM 以块长 `k` 生成草稿；LLM 在一次前向中对草稿逐 token 验证，接受前缀并在首个未通过位置采样一个自由 token，形成“draft-verify”迭代解码，见 `token_level_coinference/src/engine/hybrid_inference.py:16`（`generate`）。
- 关键步骤：
  - 草稿生成：`token_level_coinference/src/engine/hybrid_inference.py:23`（`slm.draft_generate_k`）。
  - 一次性验证：`token_level_coinference/src/engine/hybrid_inference.py:38-48`（按阈值 `p_t` 接受）。
  - 自由 token：`token_level_coinference/src/engine/hybrid_inference.py:49-52`（LLM 采样加入序列）。
- 统一 Tokenizer：SLM/LLM 共享同一 tokenizer，保证编码一致；参数可配置 `k/p_t/max_new_tokens/temperature/top_p` 等。
- 成本度量：以 `Ns/T` 与 `Nl/T` 加权统计平均成本，见 `token_level_coinference/src/engine/cost_meter.py:1` 与 `token_level_coinference/src/engine/cost_meter.py:16-23`。
- 运行模式：`slm`、`llm`、`hybrid` 三模式的统一 CLI，详见 `token_level_coinference/README.md` 与 `src/cli.py`。

## 设计理念与取舍
- 将“任务编排”与“解码算法”分治：前者面向复杂任务与长文档；后者面向生成过程本身的效率与成本。
- 本地优先、云端增强：尽量在本地完成拆解与检索，云端用于监督与高质量综合；在 token 级以 LLM 的一次性验证替代逐步校正，减少云端前向次数。
- 可插拔：模型提供方、本地/远端客户端、检索器、分块策略、量化方式均可替换，便于在不同资源环境下落地。

## 快速上手
- 任务级：阅读并按 `task_level_coinference/README.md` 配置 `Ollama` 与依赖，运行本地 RAG 示例。
- Token级：进入 `token_level_coinference`，创建虚拟环境并 `pip install -r requirements.txt`，使用 `python -m src.cli --mode hybrid --config src/config/default_config.yaml --prompts "Solve: 23+57"` 进行快速验证。
