# RSI Transfer Curriculum

用“试学后的迁移收益”训练课程生成器的可运行研究实现。项目包含：

- 固定版本的 API-Bank Agenda 工具与原始逐调用检查器。
- 严格隔离的训练、开发、测试任务生成器。
- 完整轨迹执行与终局评分。
- Qwen3 + LoRA 的 SFT 冷启动和 TRL GRPO 分支试学。
- 冻结分支、开发探针、配对迁移收益、预承诺分支推进。
- 带 KL 正则和 Adam 状态的课程分布更新。
- 逐轮 checkpoint、配置指纹和中断恢复。
- Merlin/Arnold 队列提交脚本。

## 快速验证

CPU 环境只需要 Python 3.9+ 和标准库：

```bash
export PYTHONPATH="$PWD/src"

python3 -m rsi_curriculum.vendor bootstrap
python3 -m unittest discover -s tests -v
python3 -m rsi_curriculum.smoke --output outputs/cpu_smoke.json
python3 -m rsi_curriculum.training \
  --config configs/toy.json \
  --output outputs/toy
```

`toy` 后端会跑通与真实训练相同的外循环，并真实执行 API-Bank 工具，但学习器是明确标注的确定性表格模型。它只验证编排、评分、归因和恢复，不是模型训练结果。

重复执行最后一条命令会读取 `state.json`，不会重复已提交的轮次。配置发生变化时会拒绝在原目录恢复。

## 启动 Qwen3 训练

真实后端要求 Python 3.11+、CUDA 和一张建议为 80GB 的 GPU：

```bash
python3 -m pip install -e '.[train]'
export PYTHONPATH="$PWD/src"

python3 -m rsi_curriculum.vendor bootstrap
python3 -m rsi_curriculum.training \
  --config configs/qwen3_8b_pilot.json \
  --output /persistent/path/rsi-qwen3-pilot
```

默认 pilot 使用：

- `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`
- LoRA rank 32，BF16
- 2 个试学分支，5 个外循环
- 每分支每轮 32 个训练任务、每提示 4 条 GRPO completion
- 每轮 32 个开发探针，结束后 64 个独立测试任务

这些是接口和成本 pilot 参数，不是论文主实验配置。模型与 tokenizer revision、TRL commit 和 API-Bank 源文件哈希记录在 [`upstream.lock.json`](upstream.lock.json)。

## 训练协议

每个任务要求模型一次生成由多个 JSON 对象组成的动作计划，最后提交：

```json
{"tool":"AddAgenda","arguments":{"content":"...","location":"...","time":"...","token":"..."}}
{"final":"Task completed."}
```

评分器逐个解析并执行动作。奖励由三部分组成：

- 70%：与原 API-Bank 检查器对齐的调用正确率
- 20%：完整数据库终态一致
- 10%：存在且位于最后的 `final`

严格成功还要求无缺失、错序、多余或无效动作。课程训练与分支更新使用 shaped reward；可通过 `credit_metric` 选择迁移收益使用 `reward` 或严格 `success`。

每轮流程：

1. 课程生成器为每个候选分支采样一批任务。
2. 所有分支从同一个 LoRA、优化器和调度状态开始 GRPO 更新。
3. 分支冻结后，先均匀预承诺下一轮主分支，再抽取开发探针。
4. 未更新模型与所有候选在同一探针上评分，计算配对迁移收益。
5. 用收益更新课程族概率；主学习器推进到预承诺分支，而不是事后选择最高分分支。

训练、开发和测试实体使用不同命名空间与 token；测试集只在完整运行结束后评估。逐轮报告位于 `rounds/round-*/report.json`，最终结果位于 `final_report.json`。

## Merlin 队列

提交脚本不保存内部资源标识。先从 Merlin 查询可用资源，再设置：

```bash
export MERLIN_GROUP_ID='<resource-group-id>'
export MERLIN_CLUSTER_ID='<cluster-id>'
export MERLIN_QUEUE_NAME='<queue-name>'
export MERLIN_IMAGE_URL='<cuda-pytorch-image>'

# 推荐：挂载持久卷并把输出指向其实际容器路径
export MERLIN_BYTENAS_NAME='<volume-name>'
export RSI_OUTPUT_DIR='<persistent-mounted-path>/rsi-transfer-curriculum'

# 先预览
scripts/submit_merlin.sh --dry-run

# 确认资源配置后真实提交
scripts/submit_merlin.sh
```

脚本使用 `bytedcli merlin job create-run` 提交一张 GPU 的 Arnold 任务，在容器内克隆本公开仓库、固定 Git ref、安装训练依赖、运行测试，再启动 pilot。建议提交前设置完整 commit：

```bash
export RSI_GIT_REF="$(git rev-parse HEAD)"
```

全部变量和预览模式见：

```bash
scripts/submit_merlin.sh --help
```

## 模块

| 模块 | 作用 |
| --- | --- |
| `tasks.py` | 六类 Agenda 任务、分区隔离、oracle 轨迹 |
| `trajectory.py` | 自由 completion 解析、原工具执行、完整轨迹评分 |
| `curriculum.py` | 课程采样、迁移收益策略梯度、KL 与 Adam 状态 |
| `training.py` | 分支训练、探针归因、预承诺提交、恢复与最终测试 |
| `hf_backend.py` | Qwen3、PEFT、TRL SFT/GRPO、LoRA/优化器 checkpoint |
| `toy_backend.py` | 无 GPU 的完整外循环验证 |
| `execution.py` | 分支隔离和碰撞校验的确定性工具转移缓存 |
| `sampling.py` | 全词表接受/残差采样参考实现 |

## 当前边界

- 可直接启动的是 Agenda 确定性子集的 pilot，不是完整 API-Bank。
- HF 后端已在代码层接入，但当前 Mac 没有 CUDA/训练依赖，因此尚未产生 Qwen3 训练结果。
- 工具转移缓存已用于 GRPO 奖励和跨分支探针评分。
- `sampling.py` 的精确 token 接受/残差算法有 CPU 测试，但跨 LoRA 模型的共享解码尚未接入 HF 生成路径；当前 TRL 各分支独立采样。
- 课程生成器目前学习六类可执行任务的分布，不是另一个自由生成文本的 LLM。
- 首次 GPU 运行应先做 2–5 GPU-days 的接口与成本验证，再决定是否扩展到多工具、更多分支和大规模实验。

代码库选择与已核查接口见 [`docs/CODEBASE_RESEARCH.zh.md`](docs/CODEBASE_RESEARCH.zh.md)。
