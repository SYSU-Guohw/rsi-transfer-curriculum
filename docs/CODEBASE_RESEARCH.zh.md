# 实现基础调研与选择

调研日期：2026-10-10。结论来自固定提交的源码阅读；没有运行 GPU 训练或吞吐测试。

**建议：首版采用 Transformers＋PEFT＋TRL，复用 API-Bank 的确定性工具与原始逐调用检查器，单独实现研究外循环。后续需要多卡吞吐时接入 verl。GenEnv 保留为论文基线参照，暂不直接 fork 成主实现。**

## 候选代码库

| 代码库 | 可以直接复用 | 需要自行实现或核对 | 选择 |
|---|---|---|---|
| [TRL](https://github.com/huggingface/trl) | `GRPOTrainer`、PEFT 接入、工具环境与自定义 rollout 接口、训练日志 | K 个不同学习分支、统一起点及优化器状态恢复、迁移反馈外循环；自定义 rollout API 仍属实验接口 | 首版学习器训练基础 |
| [Transformers / PEFT](https://github.com/huggingface/transformers) | Qwen3 模型前向、全词表 logits、LoRA 模型 | 私有历史、精确接受／残差采样、逐分支状态与 KV 管理 | 首版概率与模型参考实现；当前主要通过 TRL 集成源码核查 |
| [verl](https://github.com/verl-project/verl) | GRPO、FSDP、LoRA、异步 AgentLoop、vLLM/SGLang 训练推理衔接 | 多个冻结 checkpoint 的联合调度与分支隔离；全分布概率访问；版本与权重同步 | 后续规模化后端 |
| [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | 原始工具类、确定性局部状态转移、`check_api_call_correctness` | 状态快照、可评估任务变体、自由运行轨迹的评分协议 | 直接复用窄工具子集 |
| [GenEnv](https://github.com/Gen-Verse/GenEnv) | 课程数据增广框架、Agent 训练与任务生成流程 | 当前源码与论文完整双模型训练有缺口，API-Bank 适配尚需编写，依赖未锁定 | 参考基线，不作为首版主体 |

所有仓库已下载到 `.cache/upstream/` 供只读核查。具体提交和选取文件的哈希见 `upstream.lock.json`。原 ResearchStudio/IdeaSpark 研究产物不变。

## 为什么不直接改 GenEnv

已找到官方仓库，固定提交为 `30ac87276e0a13d3a84a8541365aff55bad667ad`。三个具体问题：

1. `genenv/trainer/genenv_trainer.py:96–239` 的训练循环依次训练 Agent、评估、生成任务、增广数据。环境生成器在 `:347–400` 从同一个 `env_model_path` 启动推理；在所读入口中未找到环境模型参数优化或环境 checkpoint 推进。不能把当前代码自动视为论文双模型共同训练的完整复现。
2. `genenv/utils/reward_functions.py:181–270` 的工具奖励是抽取调用后匹配名称/参数；它并不执行 API-Bank 原始工具和检查器。直接复用会改变我们的评价语义。
3. 依赖没有固定 verl 提交。该代码 `agent_trainer.fit(epochs=1)` 与本次检出的 verl `RayPPOTrainer.fit(self)` 签名不匹配；其生成 worker 还固定申请八张 GPU。直接运行需要先做兼容与资源修复。

另执行了一个窄源码测试：抽取并运行原 `_filter_prompts`，五个提示、`filtering_k=0.1` 时，`num_to_remove=0`，但 `sorted_prompts[-0:]` 取出全部提示，最终保留零条。这是实际执行的函数级缺陷，不是训练失败实验。没有修改上游代码。

因此，后续若报告 GenEnv 结果，须区分论文复现与发布代码，并逐项说明采用的语义。不能用一个缺少环境更新的弱基线支持新方法优越性。

## TRL 与 verl 的接入位置

本次 TRL 提交 `72cf37df714d3183e8e96ec391d474b26eaa30b6`：

- `trl/trainer/grpo_trainer.py:249–275,319–322`：`peft_config`、`rollout_func`、`environment_factory`。
- `:2127–2200`：自定义 rollout 返回 `prompt_ids`、`completion_ids`、`logprobs`；可返回 `env_mask` 区分模型生成与环境反馈 tokens。该版本回调需自行生成要求数量的 completions，不能沿用旧版本自动重复提示的假设。
- `:2253–2350`：逐 rollout 环境实例和 reset。`environment_factory` 需要 Transformers ≥5.2，属于实验接口。

必须区分 **K 个不同 checkpoint 的试学分支**与 **GRPO 同策略的多条采样**。不能把不同 checkpoint 混进一个 GRPO 组；外循环需要分别恢复相同起点的模型、优化器、回放和随机状态，训练各分支后冻结再评估。

本次 verl 提交 `9058412fb308e9f2e31a794ad1d18d97d4286485`：

- `verl/experimental/agent_loop/agent_loop.py`：`AgentLoopBase.run`、`AgentLoopOutput` 和 response masks。
- `tool_agent_loop.py`：多轮工具执行与 token 记录。
- `docs/advance/ppo_lora.rst`：Qwen3-8B LoRA 的 FSDP / Megatron 路线。

这些接口适合后续接入异步多卡训练。已读 rollout 输出以采样 log-probs / 可选 top-k 为主；我们的拒绝残差需要拒绝位置的完整目标与提议分布，必须另外验证获取方式与归一化，不能用 top-k 近似冒充精确采样。

## API-Bank 的真实边界

固定提交：`188835d4f9948563a6b9c8ac50cd0f3ae4021ed6`，仓库 MIT License。

- `evaluator.py:Sample.from_chat_history / Evaluator.evaluate` 评估下一次 API 调用；自然语言回复另用 ROUGE-L。它并没有现成的任意自由运行轨迹二值成功率接口。
- `ToolManager` 动态导入所有工具、使用相对工作目录，并共享可变数据库。
- `apis/__init__.py` 会遍历导入所有模块；首版只加载经检查的工具，避免无关模型、外部 API 和可变全局状态。
- 首批选择 `AddAgenda / QueryAgenda / DeleteAgenda`，以及它们的 `CheckToken` 依赖。它们只依赖显式数据库、参数和确定性日期解析；不使用当前时间或全局随机采样。
- 数据库键可能同时出现字符串和整数，查询也依赖字典迭代顺序。快照编码必须保留键类型和插入顺序，不能直接 `sort_keys=True` 后称为等价状态。

首版 smoke 数据应为明确标注的手写 fixture，不能把官方测试对话拿来训练。逐调用分数可以复用原检查器；自由运行评分与真实生成任务协议仍需独立定义并记录，不能默认视为已解决。

## 第一个代码里程碑

目标是在当前 Mac 上运行，不下载模型权重、不申请 GPU：

1. 有版本与许可证记录的 API-Bank 窄子集加载。
2. 可恢复的环境快照、私有分支历史、独立 token/action 额度、完整状态匹配的转移缓存。
3. 全词表接受／拒绝与残差采样参考实现，覆盖同分布、无重叠、零概率等情况。
4. 配对迁移反馈与归一化课程权重；未预先选定或任务不对齐时明确拒绝输入。
5. 用原始工具类运行独立执行和缓存执行的相同动作；检查结果、最终状态与逐分支额度一致。

这只证明代码组件可运行及局部规则正确，不能证明真实 Qwen3 的分布保持、实际加速、学习效果或 RSI。GPU 阶段再接 TRL rollout 与 checkpoint 事务，并记录训练、丢弃分支及评估的完整成本。

## 首个里程碑验证结果

已按上述范围完成代码，复现入口见 [README](../README.md)，保留结果见 [CPU smoke 报告](CPU_SMOKE_REPORT.json)。

- Python 3.9.6／标准库运行，32 项测试通过；未安装训练框架、下载模型权重或启动 GPU 作业。
- 使用上述原始 API-Bank 工具与检查器，三分支的 Add → Query → Delete 过程在每一步对比独立执行与缓存执行：观测、环境、历史、额度与逐调用检查结果一致。
- 手写样例中，9 个逻辑工具调用对应 4 次真实调用、5 次缓存命中。包含“私有额度分化而暂时不能复用，额度恢复一致后重新复用”的过程，不是端到端速度结果。
- 类别采样通过 225 对小离散分布的概率质量核对，并完成固定种子的采样检查；迁移反馈使用明确标注的合成分数测试预提交顺序、配对计算和权重。没有将这些数字解释成学习收益。
- 上游源文件保留原貌与 MIT License。运行时只将 `from apis.api import API` 改成包内相对导入；原文件和执行源的哈希记录在本地缓存 `provenance.json` 中。

尚未实现完整任务生成／轨迹评分协议、真实模型概率与 KV 管理、TRL 更新及 checkpoint 事务、课程生成器梯度更新和分布式后端。这些仍是下一阶段工作。
