# TrainParity 分级验收规范

> 规则：Codex 每次只完成一个 Gate。验收人运行指定命令并查看报告后，明确给出 `APPROVE GATE N`、`REWORK GATE N` 或 `STOP PROJECT`。未经批准不得进入下一 Gate。

---

## Gate 0：生态位与差异验证【一票否决】

### Codex 交付物

- `docs/COMPETITOR_ANALYSIS.md`
- `docs/PRODUCT_CONTRACT.md`
- `experiments/gate0/`
- `artifacts/gate_reports/gate_0.json`
- `artifacts/gate_reports/gate_0.md`

### 必须完成的工作

1. 安装并运行现有 OrderLab/TrainCheck；
2. 构造至少四个最小故障：
   - resume 漏 scheduler state；
   - resume 漏 RNG state；
   - gradient accumulation 的 mean-of-means；
   - distributed/evaluation sampling 的重复样本；
3. 记录现有 TrainCheck：
   - 接入步骤；
   - 是否检出；
   - 定位粒度；
   - 运行时间；
   - 误报／噪声；
4. 用不超过 100 行的 throwaway prototype 实现一个明确 A/B 比较，证明本项目的预期输出；
5. 写出不能被模糊化的产品契约和非目标。

### 机器验收

```bash
python scripts/verify_gate.py 0
```

必须确认：

- 四个故障 fixture 都可稳定复现；
- 实验日志、环境与结果完整；
- 竞品结果不是凭 README 推断，而是实际执行或明确记录无法执行的原因；
- JSON 报告 schema 合法。

### 人工验收

验收人只需回答：

1. 用户为什么不用已有 TrainCheck？
2. 本项目是否提供“显式 A/B + first divergence”这一结构性差异？
3. 这一区别是否值得用户写少量 adapter？
4. 项目名与描述是否避免混淆？

### 通过条件

- 至少两个核心故障上，本项目设计能提供现有 TrainCheck不直接提供的、明显更精确或更适合 CI 的输出；
- 差异来自执行模型，而不是 UI、命名或更漂亮的报告。

### 停止条件

- 已有 TrainCheck 以同等或更低接入成本完成相同任务；
- 差异只剩“我们预置三个 check”；
- 无法清楚解释为什么不是竞品的子功能。

---

## Gate 1：产品契约、API 原型与工程骨架

### Codex 交付物

- 可安装 Python 包骨架；
- `AGENTS.md`、`STATUS.md`、`DECISIONS.md`；
- 两种 adapter API 原型；
- 一个正确 resume case；
- 一个故障 resume case；
- lint、type-check、pytest 与 CI；
- `artifacts/gate_reports/gate_1.*`。

### API 原型要求

至少比较：

- 类／protocol 形式；
- factory + callbacks 形式。

选择标准：

- 新进程可导入；
- 类型清晰；
- 简单 case 不超过 30 行逻辑代码；
- 不依赖 cloudpickle 才能工作，除非有充分理由；
- 不要求修改被测训练代码主体。

### 机器验收

```bash
make lint
make typecheck
make test
python scripts/verify_gate.py 1
```

必须满足：

- 所有命令退出码为 0；
- CPU 环境可运行；
- 无 LLM／Agent SDK 运行时依赖；
- 包可构建 wheel；
- 简单 adapter 逻辑行数被脚本统计并写入报告。

### 人工验收

- API 是否一眼可理解；
- 用户是否知道自己必须提供什么；
- 是否诚实暴露限制；
- 是否没有提前实现平台功能。

### 停止／返工条件

- adapter 明显超过 30 行；
- 新进程无法稳健导入 case；
- API 为追求“任意 Python”而变得模糊；
- 生产依赖过多。

---

## Gate 2：核心 Snapshot、规范化与 Comparator

### Codex 交付物

- `Snapshot` schema；
- 稳定 state path；
- model/buffer/gradient 捕获；
- optimizer canonicalization；
- scheduler/scaler/extras 捕获；
- RNG 捕获；
- exact comparator；
- explicit tolerance comparator；
- first-difference report；
- 完整 unit/contract tests；
- `artifacts/gate_reports/gate_2.*`。

### 必测情况

- 嵌套 dict/list/tuple；
- tensor shape/dtype/value 差异；
- NaN、Inf、empty tensor；
- CPU/CUDA metadata；
- 参数组顺序；
- optimizer momentum/Adam states；
- 缺失 key、额外 key；
- 同值不同路径；
- 用户额外 `state_dict` 对象；
- 不可映射 optimizer 参数返回 `ABSTAIN`。

### 机器验收

```bash
python scripts/verify_gate.py 2
```

通过标准：

- fault suite 100% 产生预期的状态路径；
- clean suite 0 误报；
- exact 与 tolerance 的行为有独立测试；
- 关键核心模块测试覆盖率至少 90%；
- 任何比较失败都不能只输出二进制“不相同”。

### 人工验收

随机查看五个差异报告：

- 是否能快速看懂；
- 路径是否稳定；
- 是否避免把 first observed divergence 写成 root cause；
- optimizer 报告是否使用参数名而不是内存 ID。

---

## Gate 3：Resume Equivalence MVP【项目核心 Go/No-Go】

### Codex 交付物

- baseline 连续训练 runner；
- interrupted/save/exit/new-process/load/resume runner；
- baseline self-consistency 预检；
- `PASS / FAIL / ABSTAIN / ERROR`；
- pytest assertion；
- 文本与 JSON 报告；
- 至少十个故障注入 fixture；
- CPU 测试与可选 GPU 测试；
- `artifacts/gate_reports/gate_3.*`。

### 必测故障

1. 漏 model state；
2. 漏 optimizer state；
3. 漏 scheduler state；
4. 漏 Python RNG；
5. 漏 NumPy RNG；
6. 漏 torch CPU RNG；
7. 漏 torch CUDA RNG（GPU 环境）；
8. 漏 GradScaler；
9. 数据 cursor 偏移；
10. resume step / epoch off-by-one；
11. optimizer parameter group 错配；
12. 加载后额外 scheduler.step。

至少十项必须进入正式 suite。

### 机器验收

```bash
python scripts/verify_gate.py 3
```

通过标准：

- clean fixtures：0 误报；
- 稳定故障 fixture：100% 检出；
- 至少 80% 故障报告的首个差异组件符合预期；
- resume 前后进程 PID 不同；
- baseline 非确定时返回 `ABSTAIN`；
- 子进程异常返回 `ERROR`，不能误报为训练不等价；
- 同一 fixture 连续运行三次结果一致；
- 简单 CPU case 总执行步骤保持很小，不需要完整 epoch。

### GPU 验收

通过 Slurm 或本地 GPU 运行：

```bash
sbatch scripts/slurm_gpu_matrix.sbatch --gate 3
```

要求：

- 至少一种 GPU 完成 clean + RNG + scaler fixture；
- A100/H100/H200 之间不做相互数值比较；
- 每种 GPU 只验证该设备上的 A/B 关系。

### 人工验收

验收人运行一个命令并查看：

- clean case；
- 漏 scheduler；
- 漏 RNG；
- 数据 cursor 偏移。

报告必须能明确回答“在哪一步、哪个状态先不同”。

### 一票否决停止条件

- 只能说明最终 checkpoint 不同；
- 无法可靠启动全新 Python 进程；
- clean case 仍频繁误报；
- first divergence 与故障无关；
- adapter 复杂度使用户不如自己写测试；
- 现有 TrainCheck 在同一测试上明显更好且更简单。

---

## Gate 4：真实项目接入与用户摩擦验证【第二个 Go/No-Go】

### Codex 交付物

选择三个小型、许可清晰、可快速运行的真实项目／训练 recipe：

- 图像分类；
- 简单语言模型或序列任务；
- 一个带 scheduler 和 checkpoint 的稍复杂 loop。

每个项目提供：

- clean adapter；
- 注入故障版本；
- 接入说明；
- LOC 统计；
- 运行报告；
- 与手写测试／现有 TrainCheck 的对比；
- `artifacts/gate_reports/gate_4.*`。

### 机器验收

- 三个 clean case 全部通过；
- 三个注入故障全部检出；
- adapter 逻辑行数中位数 ≤ 30；
- 单个 case 不依赖大型数据下载；
- CI 可运行至少一个真实 case；
- 所有外部项目许可证记录完整。

### 人工验收

验收人只需：

1. 打开三个 adapter；
2. 判断它们是否是普通研究者愿意写的代码量；
3. 阅读一次差异报告；
4. 判断与直接让 Codex 临时写测试相比，是否值得保留成库。

### 停止条件

- adapter 中位数 > 30 行；
- 必须侵入式改造训练代码；
- 真实项目中 first divergence 价值不明显；
- 只能在作者自造 fixture 上工作。

---

## Gate 5：Gradient Accumulation Equivalence

### Codex 交付物

- full-batch 与 microbatch 两种 execution plan；
- 用户显式 `split_batch` 或默认 tensor split；
- 一次 optimizer update 的状态比较；
- 支持 variable-length/token-count normalization；
- 已知非等价因素提示；
- 至少七种故障 fixture；
- `artifacts/gate_reports/gate_5.*`。

### 必测故障

1. 忘记除 accumulation steps；
2. variable-length microbatch 的 mean-of-means；
3. optimizer 每个 microbatch 都 step；
4. scheduler 每个 microbatch 都 step；
5. gradient clipping 时机错误；
6. AMP unscale / scaler 时机错误；
7. 最后不足完整 accumulation window；
8. zero_grad 时机错误。

### 机器验收

- clean linear/MLP/token-level cases 通过；
- 上述稳定故障全部检出；
- BatchNorm/dropout 等案例不能被错误宣传为必然等价；
- 报告至少区分 loss、gradient、optimizer update 三层；
- tolerance 由用户显式提供；
- GPU 浮点差异不被硬编码成统一阈值。

### 人工验收

- 用户能否看出“不等价发生在 gradient 还是 optimizer step”；
- API 是否复用核心抽象，而非另造一套框架；
- 模块是否仍保持小而闭环。

---

## Gate 6：Sample Coverage Policies

### Codex 交付物

支持显式策略：

- `exactly_once`；
- `at_least_once`；
- `no_cross_rank_overlap`；
- `expected_padding`；
- 可选 epoch/shuffle consistency。

输出：

- missing IDs；
- duplicate IDs；
- 每个 ID 的 rank/worker/epoch 轨迹；
- 汇总计数；
- `artifacts/gate_reports/gate_6.*`。

### 机器验收

- world size 1/2/3/4 的 synthetic sampler fixtures；
- 数据长度不能整除 world size；
- `drop_last`；
- padding duplicate；
- 自定义 sampler；
- IterableDataset 的可控 fixture；
- 所有预期重复／遗漏均精确报告；
- 仅需 sample ID extractor，不要求读取样本内容。

### 人工验收

- 对“DistributedSampler padding 导致验证重复”的案例，报告是否直观；
- 用户是否只需提供一个简单 `sample_id`；
- 模块是否没有扩成 DDP 训练框架。

### 停止条件

- 用户必须大改 dataset；
- 输出只是一堆日志，无法给出确定集合结论；
- 功能简单到一段十行脚本已经同样好用，且没有复用核心价值。

---

## Gate 7：公共发布候选

### Codex 交付物

- 稳定包名（重新检查 PyPI/GitHub 冲突）；
- README、API 文档、限制说明；
- 三个最小示例；
- wheel/sdist；
- CPU CI；
- GPU Slurm 验证脚本；
- 许可证、贡献指南、安全说明；
- benchmark/fault matrix；
- changelog；
- release notes；
- `artifacts/gate_reports/gate_7.*`。

### 机器验收

```bash
make release-check
python scripts/verify_gate.py 7
```

必须满足：

- clean clone 可安装；
- wheel 可导入；
- README 示例可执行；
- lint/type/test 全通过；
- 无隐藏 API key 或在线依赖；
- 依赖许可证检查通过；
- public API 有测试；
- 包名和 import name 明确；
- 报告中给出版本、Python、PyTorch、CUDA、GPU 元数据。

### 人工验收

- README 是否清楚说明与 OrderLab/TrainCheck 的差异；
- 是否避免夸大支持范围；
- 是否不需要 Web 服务或持续维护平台；
- 用户能否在几分钟内理解并运行第一个 case；
- 是否值得公开，而不是仅留作内部测试代码。

### 发布动作

Codex 只准备发布。真正创建远程仓库、发布 PyPI、推送 tag 等写操作由验收人明确批准后执行。

---

## Gate 8+：条件扩展，不属于当前承诺

候选项：

- AMP/GradScaler 完整支持；
- multi-worker StatefulDataLoader；
- phase-level hooks；
- DDP；
- Lightning adapter；
- Transformers Trainer adapter；
- CI cache 与大型状态 fingerprint；
- property-based fault generation。

每项必须先提交独立提案，包括：

1. 用户需求证据；
2. 与现有工具的差异；
3. 维护成本；
4. 自动验收；
5. 不做时的替代方法；
6. 对核心 API 的影响。

未获人工批准不得实施。
