# TrainParity 项目总体规划书

> 工作名称：**TrainParity**  
> 项目性质：面向 PyTorch 训练流程的差分／等价性测试库  
> 规划版本：v0.1  
> 日期：2026-08-10  
> 本文用途：作为 Codex Goal Mode 的主实施规范；开发者按验收门逐级实施，验收人只在每个 Gate 结束时复核并决定继续、返工或停止。

---

## 0. 立项前提与重要纠偏

### 0.1 不能再使用 “TrainCheck” 作为项目名或宽泛定位

目前已经存在 OrderLab/TrainCheck：一个 OSDI 2025 项目，通过追踪 PyTorch API 与模型状态、从健康运行中推断不变量，再检测目标训练中的静默错误。它已经开源、发布 PyPI 包并持续维护。

因此，本项目不得：

- 使用 `TrainCheck` 名称；
- 宣称“自动检测任意训练错误”；
- 复制“健康运行 → 推断不变量 → 检查目标运行”的工作流；
- 以“训练可观测性平台”或“通用训练医生”作为定位；
- 在没有差异验证的情况下直接进入实现。

### 0.2 本项目唯一允许的核心定位

> **TrainParity 验证用户明确声明的两种 PyTorch 训练执行是否满足等价关系，并定位第一个可观测状态分叉。**

核心不是“推断正确行为”，而是由用户声明一个可测试的关系：

- 连续训练，是否等价于保存、退出进程、加载后继续训练；
- 一个完整 batch 的更新，是否等价于若干 microbatch 的梯度累积；
- 一次数据遍历，是否满足 exactly-once、无跨 rank 重叠或其他显式覆盖策略。

### 0.3 项目成立的必要差异

| 维度 | 已有 OrderLab/TrainCheck | 本项目 TrainParity |
|---|---|---|
| 正确性来源 | 从参考运行推断行为不变量 | 用户显式声明 A/B 等价关系 |
| 工作流 | 收集参考 trace、推断、检查目标 trace | 从同一初始条件构造两个受控执行并直接比较 |
| 主要用途 | 广泛捕捉静默训练错误 | 针对训练语义变换做可重复回归测试 |
| 核心输出 | 违反了哪些推断不变量 | 第一个分叉 step、状态路径与数值差异 |
| 接入目标 | 尽量不改训练代码、运行时追踪 | 少量显式 adapter，换取可控和可验证 |
| 产品形态 | 训练可观测与主动检查框架 | 类似 pytest 的训练语义测试库 |

**Gate 0 必须实测这一差异。若差异不能成立，项目立即终止。**

---

## 1. 一句话产品定义

> **TrainParity is a differential-testing library for PyTorch training semantics: declare two executions that should be equivalent, run them under controlled conditions, and locate their first observable divergence.**

中文表述：

> 用户声明两种训练方式理论上应该等价；TrainParity 在隔离进程中真实运行两条轨迹，检查 batch、RNG、模型、梯度、优化器、调度器及用户注册状态，并报告最早从哪里开始不同。

---

## 2. 目标用户、实际需求与使用时机

### 2.1 第一目标用户

1. 维护自定义 PyTorch training loop、recipe 或 trainer 的研究工程师；
2. 修改 checkpoint、resume、gradient accumulation、sampler、AMP 逻辑的框架开发者；
3. 需要在 Slurm、抢占式任务或长时间训练中可靠恢复的科研团队；
4. 在发布训练代码前希望加入语义回归测试的开源仓库维护者。

### 2.2 用户不会每天运行它

本工具不是训练监控器。典型使用时机是：

- 修改 checkpoint 格式或恢复逻辑后；
- 更换 optimizer、scheduler、GradScaler 或 DataLoader 后；
- 实现梯度累积或可变长度 loss 后；
- 从单卡切换到新的采样／数据切分方式后；
- 发布新版本或合并相关 PR 前；
- 开始高成本长训练前。

### 2.3 用户的直接收益

- 防止“任务能继续跑，但恢复后的训练轨迹已经变了”的静默错误；
- 将一次性的调试脚本变成可长期保留的 CI 测试；
- 比较的不只是最终 loss，而是第一个分叉的 step 与状态路径；
- 不需要 Codex、API key、云服务或后台平台；
- 不要求改用新的训练框架。

---

## 3. 为什么不是“直接让 Codex 检查”

Codex 可以阅读 checkpoint 代码并发现明显漏项，也可以临时编写比较脚本。但 TrainParity 的独立价值必须来自以下确定性能力：

1. 从同一初始条件构造两条隔离执行；
2. 强制 resume 分支真正退出并在新 Python 进程加载；
3. 捕获标准 PyTorch 状态及用户注册状态；
4. 对状态做稳定、可重放、路径级比较；
5. 先做 baseline 自一致性测试，无法比较时返回 `ABSTAIN`；
6. 将测试永久集成到 pytest/CI；
7. 对多个故障注入案例保持固定验收结果。

**用户运行 TrainParity 时，不调用任何 LLM。**  
Codex 只用于开发本项目，或帮助用户为自己的仓库生成 adapter。

---

## 4. 项目边界

### 4.1 核心范围

- Python + PyTorch；
- 用户显式提供小型 `Case` adapter；
- 受控的 A/B differential execution；
- 单进程、单 GPU 为首版主范围；
- 标准 PyTorch 状态自动捕获；
- 用户自定义状态注册；
- `PASS / FAIL / ABSTAIN / ERROR` 四态结果；
- pytest 优先，CLI 作为薄封装；
- 文本与 JSON 报告；
- Slurm 仅用于开发测试，不成为运行时依赖。

### 4.2 首版明确不做

- 自动理解任意 `train.py`；
- 自动寻找所有训练 bug；
- 基于 LLM 的根因诊断；
- 训练监控 dashboard；
- Web 服务、SaaS、registry 或 leaderboard；
- 自动调参、显存估算、训练加速；
- 跨 GPU 型号 bitwise 对比；
- DDP、FSDP、ZeRO、Pipeline Parallel、Tensor Parallel；
- Lightning、Transformers、DeepSpeed 的正式 adapter；
- 长期收集用户训练数据；
- 自动判定模型最终是否“训练得好”。

### 4.3 扩展原则

任何新增功能必须同时满足：

1. 属于“用户声明的不变量／等价关系”；
2. 可自动构造 verifier；
3. 不要求持续运营；
4. 不把项目变成训练平台；
5. 至少有三个独立真实需求证据；
6. 单独通过新的 Gate 方案与人工批准。

---

## 5. 结果语义

TrainParity 必须严格区分以下结果：

### `PASS`

- baseline 自一致；
- candidate 与 baseline 在指定比较策略下等价；
- 所有必要观测均成功取得。

### `FAIL`

- baseline 自一致；
- candidate 在某个 step／状态路径首次超出比较策略；
- 报告中有可重现证据。

### `ABSTAIN`

TrainParity 无法可靠判断，例如：

- baseline 连续运行两次自身就不一致；
- 用户要求 exact，但使用了已知非确定性操作；
- 无法取得必要状态；
- adapter 未提供 sample ID 或 checkpoint 恢复接口；
- 当前功能不支持该训练结构。

`ABSTAIN` 不能计为失败，也不能伪装成通过。

### `ERROR`

测试基础设施本身失败，例如：

- 子进程崩溃；
- checkpoint 文件无法读取；
- adapter 接口违反协议；
- 序列化、临时目录或环境错误。

---

## 6. 目标用户接口

### 6.1 主接口：pytest

目标体验：

```python
# tests/parity_cases.py
from trainparity import TorchCase, TorchState

class TinyClassifierCase(TorchCase):
    def build(self, seed: int) -> TorchState:
        ...

    def next_batch(self, state: TorchState):
        ...

    def train_step(self, state: TorchState, batch):
        ...

    def save(self, state: TorchState, path):
        ...

    def load(self, path, seed: int) -> TorchState:
        ...
```

```python
# tests/test_training_parity.py
from trainparity import assert_resume_equivalent

def test_resume_equivalence():
    assert_resume_equivalent(
        case="tests.parity_cases:TinyClassifierCase",
        split_step=4,
        steps_after_resume=3,
        comparison="exact",
    )
```

运行：

```bash
pytest tests/test_training_parity.py -q
```

### 6.2 梯度累积接口

```python
from trainparity import assert_accumulation_equivalent

def test_accumulation_equivalence():
    assert_accumulation_equivalent(
        case="tests.parity_cases:VariableLengthLMCase",
        microbatches=4,
        comparison={"rtol": 1e-6, "atol": 1e-8},
    )
```

### 6.3 数据覆盖接口

```python
from trainparity import assert_sample_policy

def test_validation_samples_exactly_once():
    assert_sample_policy(
        case="tests.parity_cases:ValidationLoaderCase",
        world_size=4,
        epochs=1,
        policy="exactly_once",
    )
```

### 6.4 CLI

CLI 只作为 pytest/API 的薄封装：

```bash
trainparity resume tests.parity_cases:TinyClassifierCase \
  --split-step 4 \
  --steps-after-resume 3 \
  --comparison exact
```

### 6.5 接入成本硬指标

对于一个标准的单卡 PyTorch 训练循环：

- adapter 目标不超过 30 行逻辑代码；
- 不要求修改模型定义；
- 不要求改用 TrainParity 自己的 Trainer；
- 不要求在生产训练中常驻 instrumentation。

若真实仓库普遍需要 50 行以上 adapter，必须暂停并重新评估产品价值。

---

## 7. 核心技术模型

### 7.1 统一抽象：Execution Plan

每个 check 生成两个执行计划：

```text
baseline plan  A
candidate plan B
```

二者从同一逻辑初始条件出发，但执行方式不同。

### 7.2 Trajectory

每条执行产生一串快照：

```text
Run metadata
Step 0 snapshot
Step 1 snapshot
...
Step N snapshot
```

MVP 只保证 step-boundary 定位。阶段级定位属于后续扩展。

### 7.3 Snapshot

标准快照包括：

- batch fingerprint / sample IDs；
- Python RNG；
- NumPy RNG（若安装）；
- PyTorch CPU RNG；
- PyTorch CUDA RNG；
- model parameters；
- model buffers；
- named gradients；
- optimizer state；
- scheduler state；
- GradScaler state；
- learning rate；
- loss／用户注册 metric；
- 用户额外注册的 `state_dict` 状态。

### 7.4 State path

所有差异必须使用稳定路径表达，例如：

```text
model.encoder.layers.3.weight
gradient.lm_head.weight
optimizer.param_groups.0.lr
optimizer.state.encoder.weight.exp_avg
scheduler.last_epoch
rng.torch_cuda.device_0
batch.sample_ids
extra.ema.shadow.encoder.weight
```

### 7.5 Optimizer canonicalization

PyTorch optimizer 的原生 state 可能依赖参数 ID 与参数组顺序。TrainParity 必须：

1. 通过 `model.named_parameters()` 建立参数名映射；
2. 将 optimizer state 规范化为基于参数名的稳定表示；
3. 检测参数别名、重复参数或不可映射状态；
4. 无法可靠规范化时返回 `ABSTAIN`，不得用不稳定 ID 比较。

### 7.6 比较策略

首版支持：

- `ExactComparison`；
- `ToleranceComparison(rtol, atol, equal_nan)`；
- 非 tensor 标量、字符串、列表、字典的递归比较；
- shape、dtype、device metadata；
- NaN、Inf 与 signed zero 的明确策略。

禁止在首版自动“猜”容差。用户必须显式指定，或使用 exact。

### 7.7 First divergence

算法输出：

- 最早发生差异的 step；
- 同一步所有首次差异路径；
- baseline 与 candidate 的摘要；
- 最大绝对／相对误差；
- 差异出现前最后一个相同快照；
- 后续差异只作为 downstream 信息，不伪称根因。

首版输出的是“first observed divergence”，不是“真实根因”。

---

## 8. 推荐仓库结构

```text
trainparity/
├── AGENTS.md
├── PLAN.md
├── ACCEPTANCE.md
├── CODEX_GOALS.md
├── STATUS.md
├── DECISIONS.md
├── CHANGELOG.md
├── LICENSE
├── README.md
├── pyproject.toml
├── Makefile
├── src/
│   └── trainparity/
│       ├── __init__.py
│       ├── protocols.py
│       ├── result.py
│       ├── state.py
│       ├── snapshot.py
│       ├── rng.py
│       ├── optimizer_state.py
│       ├── comparison.py
│       ├── trajectory.py
│       ├── divergence.py
│       ├── subprocess_runner.py
│       ├── report.py
│       ├── pytest_plugin.py
│       ├── cli.py
│       └── checks/
│           ├── resume.py
│           ├── accumulation.py
│           └── sampling.py
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── faults/
│   ├── integration/
│   └── gpu/
├── examples/
│   ├── resume/
│   ├── accumulation/
│   └── sampling/
├── scripts/
│   ├── verify_gate.py
│   └── slurm_gpu_matrix.sbatch
└── artifacts/
    └── gate_reports/
```

---

## 9. 工程规范

建议工具链：

- 包管理：`uv` 或标准 `pip`，由 Gate 1 决定；
- 格式与 lint：`ruff`；
- 类型检查：`pyright` 或 `mypy`，二选一；
- 测试：`pytest`；
- 覆盖率：`coverage.py` / `pytest-cov`；
- 文档：首版只用 Markdown，不搭建文档站；
- CI：GitHub Actions 的 CPU 测试；
- GPU：通过本地／Slurm 脚本手动或定期执行；
- 生产依赖尽量仅包含 PyTorch 与少量标准库辅助包；
- 不引入数据库、Web 框架、Agent SDK 或 LLM API。

每个 Gate 必须产出：

```text
artifacts/gate_reports/gate_<N>.json
artifacts/gate_reports/gate_<N>.md
```

JSON 用于机器验收；Markdown 用于人工验收。

---

## 10. 分层功能路线

### Level A：可行性与差异验证

只验证项目是否值得做，不写完整工具。

### Level B：核心差分引擎

实现状态捕获、规范化、比较、结果类型与报告。

### Level C：Resume MVP

这是项目的主价值验证。未通过前不得开发其他模块。

### Level D：真实仓库接入

验证用户 adapter 成本和实际定位价值。

### Level E：Gradient Accumulation

在核心引擎稳定后增加第二个显式等价关系。

### Level F：Sample Coverage

增加集合／覆盖型不变量，但不扩成分布式训练平台。

### Level G：公共发布

完善稳定 API、文档、wheel、CI 与安全边界。

### Level H：条件扩展

AMP、多 worker、DDP 等必须另立方案，不自动进入。

---

## 11. Codex 实施原则

1. 每次只执行一个 Gate；
2. Gate 内可以自主迭代，Gate 之间必须暂停等待人工批准；
3. 任何验收失败先修复，不得绕过测试继续；
4. 不允许为了通过测试而削弱 predicate、删除故障 fixture 或放宽默认容差；
5. 不允许实现本文明确排除的功能；
6. 不允许把 OrderLab/TrainCheck 代码复制进本项目；
7. 所有第三方代码与依赖必须记录许可证和理由；
8. 每个 Gate 开始前更新 `STATUS.md`；
9. 每个 Gate 结束时生成机器报告、人工摘要和 Git diff 摘要；
10. 无法满足某个验收条件时，明确报告 `BLOCKED`，不得伪造通过；
11. 使用 Git checkpoint，确保每个 Gate 可回滚；
12. Codex 可以使用只读 subagent 做竞品分析或代码审查，但不要让多个写入 agent 同时修改同一工作树。

---

## 12. 风险与停止条件

### 风险 1：与已有 TrainCheck 重合

**停止条件：**现有 TrainCheck 在同样或更少接入成本下，已能可靠完成 resume／accumulation／sampling 的显式等价性检查和 first-divergence 定位。

### 风险 2：adapter 成本过高

**停止条件：**三个真实项目的 adapter 中位数超过 30 行逻辑代码，或必须侵入性修改训练框架。

### 风险 3：baseline 自身不稳定

**策略：**返回 `ABSTAIN`；不扩大 claim。  
**停止条件：**大部分目标训练场景都无法通过自一致性预检，使工具只能用于玩具案例。

### 风险 4：只会报告“最终不同”

**停止条件：**无法稳定定位到第一个 step 和具体状态路径，价值不足以超过用户自己比较 checkpoint。

### 风险 5：数值误报

**策略：**exact 与 tolerance 明确分离，不自动猜容差。  
**停止条件：**clean fixtures 或真实 clean 项目出现不可解释的高误报。

### 风险 6：scope 膨胀

**停止条件：**项目开始依赖 Web 服务、模型 registry、十余个 framework adapter 或持续运营，立即回退。

### 风险 7：算力被硬凑进项目

GPU 只用于真实执行验证，不是卖点。不得为了使用 H100/H200 而添加无关 benchmark 或大模型训练。

---

## 13. 最终成功标准

项目只有同时满足以下条件，才值得公开发布：

1. 与 OrderLab/TrainCheck 有经过实测的结构性差异；
2. 用户运行时不需要 LLM；
3. Resume 模块对预设故障检测率达到验收标准；
4. clean fixtures 无误报；
5. 能报告 first observed divergence；
6. 真实项目 adapter 成本可接受；
7. 三个模块共享同一个小型核心，而不是拼装平台；
8. 核心测试在 CPU CI 中可运行，GPU 检查有独立脚本；
9. README 不作“适用于任意 Python 训练”的虚假承诺；
10. 无服务器、无数据库、无长期运营要求。

---

## 14. 对外定位草案

### README 标题

**TrainParity — Differential tests for PyTorch training semantics**

### 一句话说明

> Verify that resume, gradient accumulation, and data sampling preserve the training semantics you intended.

### 不应出现的宣传语

- “Finds every training bug”
- “Works on any Python training script”
- “AI training doctor”
- “Universal training observability”
- “One-click debugging”
- “Drop-in replacement for your trainer”

---

## 15. 验收流程总览

```text
Gate 0 竞品与生态位验证
   ↓ 人工批准
Gate 1 产品契约与 API 原型
   ↓ 人工批准
Gate 2 核心状态比较引擎
   ↓ 人工批准
Gate 3 Resume MVP
   ↓ 一票否决 Go/No-Go
Gate 4 真实项目接入
   ↓ 一票否决 Go/No-Go
Gate 5 Gradient Accumulation
   ↓ 人工批准
Gate 6 Sample Coverage
   ↓ 人工批准
Gate 7 发布候选
   ↓ 最终验收
Gate 8+ 条件扩展（另行立项）
```

详细条件见 `ACCEPTANCE.md`。Codex 的逐 Gate 目标提示见 `CODEX_GOALS.md`。
