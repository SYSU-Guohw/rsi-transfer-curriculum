# RSI Transfer Curriculum

基于“试学后的迁移收益训练课程生成器”研究提案的首个 CPU 原型。

**代码选择：Transformers＋PEFT＋TRL 作为后续学习器后端，API-Bank 提供工具，研究外循环独立实现。** 多卡阶段再评估 verl。GenEnv 作为需要核对论文／代码差异的基线，当前不直接 fork。理由、源码位置和固定提交见[代码库调研](docs/CODEBASE_RESEARCH.zh.md)与[版本锁](upstream.lock.json)。

当前实现不依赖 torch，未接入模型训练。Python 3.9+，运行与测试只用标准库。

## 运行

在本项目目录执行：

```bash
export PYTHONPATH="$PWD/src"

# 下载固定提交的 5 个 Python 源文件和 MIT LICENSE，并逐文件校验 SHA-256。
# 不下载模型、数据集或训练框架。
python3 -m rsi_curriculum.vendor bootstrap

python3 -m unittest discover -s tests -v
python3 -m rsi_curriculum.smoke --output outputs/cpu_smoke.json
```

当前工作区已有上游 checkout，也可离线初始化：

```bash
python3 -m rsi_curriculum.vendor bootstrap --checkout .cache/upstream/damo-convai
```

bootstrap 后，测试和 smoke 全程离线。可选以 `python3 -m pip install -e .` 在自己的虚拟环境中进行可编辑安装；当前复现实验使用上面的 `PYTHONPATH` 命令，无需安装包。

## 已实现

| 模块 | 能力 |
| --- | --- |
| [vendor.py](src/rsi_curriculum/vendor.py) | 固定版本、哈希验证、保留许可证；只加载 API／CheckToken／三个 Agenda 工具，绕开上游全量动态导入 |
| [state.py](src/rsi_curriculum/state.py) | 保留键类型和字典插入顺序的字节快照；显式拒绝未知对象、可变别名和循环 |
| [environment.py](src/rsi_curriculum/environment.py) | 独立环境恢复；分支内工具共享 Agenda／Account 数据库；复制 QueryAgenda 返回的可变观测 |
| [execution.py](src/rsi_curriculum/execution.py) | 独立执行与完整状态／调用缓存；哈希命中后比较原始字节；私有历史、token/action 额度和逐调用检查结果分别更新 |
| [sampling.py](src/rsi_curriculum/sampling.py) | 全类别概率向量上的接受／拒绝、正残差采样与稳定 softmax；处理零概率及同分布情形 |
| [credit.py](src/rsi_curriculum/credit.py) | 先均匀选择提交分支、后请求开发探针；严格对齐所有分支的探针，计算配对增益及指数归一化权重 |
| [fixtures.py](src/rsi_curriculum/fixtures.py)、[smoke.py](src/rsi_curriculum/smoke.py) | 手写样例，执行原始工具并比较两种执行方式；类别采样检查与合成分数算术检查 |

2026-10-10 在当前 Mac／Python 3.9.6 验证：**32 项测试通过**。保留的[CPU smoke 报告](docs/CPU_SMOKE_REPORT.json)记录：

- 三个分支完成 Add → Query → Delete，9 个逻辑调用的逐步观测、私有历史、状态和额度与独立执行一致；原始逐调用检查全部通过。
- 缓存共命中 5 次，实际调用由 9 次变成 4 次。这是手写高重合样例中的调用次数，不是耗时或训练加速测量。
- 分支 token 额度曾为 `[26, 24, 26]`，因此中间分支不能复用下一步；后续额度恢复一致才再次命中。不同历史始终保留。
- 测试覆盖强制哈希碰撞、隐藏 Account 状态、混合键／顺序、跨分支写隔离、失败工具、无效与截断动作、源文件篡改，以及 225 对小离散分布的概率质量核对。

## 当前执行协议

这是本原型的明确约定，**不是 API-Bank 官方自由运行轨迹评分协议**：

```json
{"tool":"AddAgenda","arguments":{"token":"synthetic-demo-token","content":"write meeting notes","time":"2025-01-02 09:00:00","location":"Room A"}}
```

最终回复使用 `{"final":"..."}`。解析只归一化 JSON 空白和参数排列，不改参数内容；重复键、未知工具、多余参数或类型错误均判为无效动作。

调用者提供该分支实际生成的 token 数；smoke 使用人为指定的测试计数。完整动作（含无效动作和 final）扣一次 action 及自身 token；token 用尽前未完成的动作只扣 token、不执行工具。恰好在 token 上限内完成的调用先执行再终止。终止分支不能重启或加入复用。

缓存键包含适配器／上游版本、任务 ID、Account／Agenda 全部状态、额度、参考调用、检查器游标与结果、终止状态和规范调用。缓存值只有环境效果和观测，不包含代表分支的记账。工具不读取私有文本历史，因此历史不进入工具缓存键；它始终保留在各分支中。若扩展为读取历史的工具、解释器、时钟、RNG 或检查器，必须扩展状态契约再启用复用。

参考检查仅按 fixture 顺序调用原始 `check_api_call_correctness`，不汇总成任务成功率，不评价自然语言答案。没有使用官方测试样本作训练或 smoke 输入。

采样模块处理调用者提供的完整分布，使用 Python 浮点数。它不能判定上游是否删掉了词表概率，也尚未实现私有模型前缀打分、token 块验证或 KV 回滚。`CreditRound` 的冻结分支 ID 是记录标签，不替代模型／优化器快照，也未执行生成器梯度更新。

## 接下来接入训练的顺序

1. **先补数据与评分契约。** 定义生成任务 → 初始状态 → 原始调用参考的合法变换、训练／开发／最终测试划分，以及完整轨迹到逐调用评分的映射。尤其明确遗漏、多余调用、截断及自然语言答案的处理。
2. **接 Transformers 概率参考实现。** 固定 Qwen3 revision／tokenizer 与温度，获取完整 logits；验证每个分支自己的前缀、拒绝后独立续写、真实 token 计数和 KV 回滚。先跑独立执行，再加 token 推测和缓存。
3. **接 PEFT＋TRL 分支试学。** 实现共同模型／优化器／scheduler／随机数／回放快照，分别 GRPO 更新并冻结，再抽探针；按已预选分支提交全部训练状态。生成器使用 detached 权重及初始生成器 KL 更新。
4. **做成本 pilot 与对照。** 同一后端比较独立执行、共同随机数、仅 token 推测、仅缓存及普通组合；计入全部训练分支、验证、残差解码、状态处理、工具和生成器成本，再决定是否迁移 verl。

目前交付的是可运行的研究组件与验证基座。没有 Qwen3 训练结果、完整 RSI 闭环、官方任务成功率、新颖性结论或实测 GPU 资源估计。
