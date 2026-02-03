# Pruning 模型压缩方案与实施计划

基于当前 KD 项目（ResNet-34 / MobileNetV2，CIFAR / TinyImageNet），下面是剪枝方法推荐与分阶段实施计划。

---

## 一、剪枝方法推荐

### 1. 按剪枝粒度

| 类型 | 说明 | 推荐度 | 备注 |
|------|------|--------|------|
| **结构化剪枝** (Structured) | 按通道/卷积核整块移除 | ⭐⭐⭐ 优先 | 实际推理加速明显，易部署 |
| **非结构化剪枝** (Unstructured) | 逐权重置零 | ⭐⭐ | 需稀疏内核/库才有加速，适合研究 |

**建议**：以**结构化剪枝**为主（channel/filter pruning），方便和你现有的 CNN 对接，且在 CPU/GPU 上都能拿到真实加速。

### 2. 具体方法推荐

| 方法 | 库/实现 | 适合场景 | 与 KD 结合 |
|------|---------|----------|------------|
| **L1-norm channel pruning** | `torch-pruning` | 通用、稳定 | 对 teacher 或 student 剪枝后蒸馏 |
| **BN scaling / filter importance** | 自实现 或 `torch-pruning` |  ResNet / MobileNet | 用 BN γ 做重要性，再剪通道 |
| **Gradual magnitude pruning** | `torch.nn.utils.prune` | 快速验证 | 非结构化，先跑通流程 |
| **OBS / OBD** | 自实现 或 专用库 | 进阶、二阶信息 | 精度略好，实现复杂 |

**首选**：**L1-norm channel pruning**（如 `torch-pruning`）+ **gradual pruning**（训练过程中逐步提高稀疏度），实现简单、可复现、易和你现有 `train_teacher` / `train_student` 对接。

### 3. 库推荐

- **[torch-pruning](https://github.com/VainF/Torch-Pruning)**（推荐）
  - 结构化剪枝、依赖感知、支持 ResNet / MobileNet 等
  - API 清晰，便于写 `prune_model(model, example_input, amount=0.3)` 这类封装
- **PyTorch 内置** `torch.nn.utils.prune`
  - 适合先做 **unstructured** 小实验（如 `l1_unstructured`）
  - 无需新依赖，验证「训练 + 剪枝 + 评估」管线
- **nncf**（Intel）
  - 若需要 quantization + pruning 一体化、部署到 Intel 硬件时再考虑

---

## 二、与 KD 的结合方式

三种典型用法，按实现难度排序：

1. **Prune-then-Distill**
   - 先训练 teacher → 对 teacher 做结构化剪枝 → 用剪枝后的 teacher 蒸馏 student。
   - 优点：实现最简单，只改 teacher 导出与 student 的 teacher 加载。
   - 适合：验证「小 teacher + KD」是否比「大 teacher + KD」更高效。

2. **Distill-then-Prune**
   - 先正常 KD 训练 student → 再对 student 做剪枝（可加若干 epoch 微调）。
   - 优点：不碰 teacher，只压缩 student，和现有 `train_student` 流程接近。
   - 适合：目标是把 **student 压到更小、更快**。

3. **Prune + Distill 联合**
   - 在 student 训练过程中同时做渐进式剪枝（例如每 N 个 epoch 提高一点剪枝率）。
   - 优点：压缩与蒸馏一步完成，有机会更好平衡精度与大小。
   - 适合：作为进阶阶段，在 1、2 跑通后再做。

建议实施顺序：**先 2（Distill-then-Prune），再 1（Prune-then-Distill），最后 3（联合）**。

---

## 三、分阶段实施计划

### Phase 1：环境与最小验证（约 1–2 天）

- [x] **1.1** 安装依赖  
  - `pip install torch-pruning thop`（可选，用于结构化剪枝与 FLOPs）；  
  - 仅用 `torch.nn.utils.prune` 也可完成 Phase 1。
- [x] **1.2** 写一个 **standalone 剪枝脚本**（如 `src/prune_utils.py`）：
  - 加载已有 `checkpoints`（teacher 或 student）；
  - 对 `nn.Linear` / `nn.Conv2d` 做 **L1 unstructured** 剪枝（例如 30%），用 `torch.nn.utils.prune`；
  - 保存剪枝后模型，并跑 `evaluate` 逻辑看精度变化。
- [x] **1.3** 在 CIFAR-10 上跑通：**正常 KD student → 剪枝 student → 评估**，确认流程没问题。

**产出**：`prune_utils.py`（含 load → prune → save）、`prune_student.py` 流水线、可复现的「Distill-then-Prune」实验（含准确率前后对比）。

---

### Phase 2：结构化剪枝（Channel Pruning）（约 3–5 天）

- [x] **2.1** 引入 `torch-pruning`，实现 **channel-level 剪枝**：
  - 封装 `apply_structured_prune(model, example_inputs, amount, ignored_layers, ...)`；
  - 对 **MobileNetV2 student** 做 L2-norm channel 剪枝，忽略最终 classifier。
- [x] **2.2** 做 **Distill-then-Prune** 实验：
  - 用现有 `train_student` 训练 student；
  - `prune_student.py` 支持对 student 做不同剪枝率，评估 **参数量、FLOPs、accuracy、inference time**；
  - 记录到 `checkpoints/pruning_comparison.csv`，便于和 `distill_method_comparison` 对比。
- [ ] **2.3**（可选）对 **ResNet-34 teacher** 做 channel 剪枝，再蒸馏：
  - 实现 **Prune-then-Distill**；
  - 对比「原始 teacher + KD」vs「剪枝 teacher + KD」的 student 精度与训练成本。

**产出**：结构化剪枝工具函数、Distill-then-Prune 流水线与脚本、`pruning_comparison.csv`。

---

### Phase 3：渐进式剪枝与超参（约 2–3 天）

- [ ] **3.1** **Gradual pruning**：
  - 在 `train_student` 的每个 epoch 结束时，对 student 的部分层做小幅剪枝（如每 epoch 增加 1% 稀疏度，直到目标）；
  - 可用 `torch-pruning` 的 scheduler 或自写简单 schedule。
- [ ] **3.2** 超参数搜索（在小规模设置下）：
  - 剪枝率、gradual 的 schedule、是否微调、微调 epoch 数；
  - 记录最佳配置，写入 `configs/config.yaml` 或单独 `configs/pruning.yaml`。
- [ ] **3.3** 若有时间，尝试 **Prune + Distill 联合**：
  - 训练 loop 里同时做 KD loss 与渐进剪枝；
  - 与 Phase 2 的 Distill-then-Prune 对比精度与模型大小。

**产出**：Gradual pruning 实现、超参建议、以及（若完成）联合实验对比。

---

### Phase 4：复现与文档（约 1–2 天）

- [x] **4.1** 添加 **scripts**：
  - `run_prune_student.sh`，与现有 `run_distill_methods.sh` 风格统一；
  - 支持环境变量 `STUDENT_CKPT`、`PRUNE_RATIO`、`PRUNE_METHOD`、`FINETUNE_EPOCHS` 等； CLI 也可直接指定。
- [ ] **4.2** 更新 **README**：
  - 如何安装额外依赖、如何运行剪枝实验、如何解读 `results/` 里的表格；
- [ ] **4.3** 在 `results/` 下写简短 **PRUNING_REPORT.md**：
  - 剪枝设置、主要结果（参数量、FLOPs、accuracy）、与现有 KD 的对比、结论与后续可做工作。

**产出**：可复现脚本、README 更新、PRUNING_REPORT.md。

---

## 四、建议的代码结构

```text
knowledge-distillation/
├── configs/
│   ├── config.yaml
│   └── pruning.yaml          # 新增：剪枝相关配置
├── src/
│   ├── prune_utils.py        # 新增：剪枝工具（unstructured + structured 封装）
│   ├── train_student.py      # 后续可加 --prune 等逻辑
│   └── ...
├── scripts/
│   ├── run_prune_student.sh           # Distill-then-Prune
│   ├── run_prune_teacher_distill.sh   # Prune-then-Distill
│   └── ...
├── results/
│   ├── distill_method_comparison.csv
│   ├── pruning_comparison.csv         # 新增：剪枝率 vs 精度/大小
│   └── PRUNING_REPORT.md              # 新增
└── docs/
    └── PRUNING_PLAN.md                # 本文件
```

---

## 五、依赖建议

当前 `requirements.txt` 基础上，建议新增：

```text
# Pruning (optional, for Phase 2+)
torch-pruning>=1.3.0
```

Phase 1 仅用 `torch.nn.utils.prune` 时可不加。

---

## 六、简要总结

| 阶段 | 内容 | 预期产出 |
|------|------|----------|
| **Phase 1** | 内置 prune、Distill-then-Prune 小实验 | `prune_utils`、流程跑通 |
| **Phase 2** | 结构化 channel 剪枝、Distill / Prune 两个顺序 | 剪枝率–精度–大小对比 |
| **Phase 3** | Gradual pruning、联合 Prune+KD、超参 | 更优压缩与配置 |
| **Phase 4** | 脚本、README、报告 | 可复现与文档 |

优先完成 **Phase 1 + Phase 2**，即可得到一批有参考价值的剪枝+KD 结果；Phase 3–4 视时间再推进。若你愿意，我可以从 Phase 1 的 `prune_utils.py` 和 `run_prune_student` 的具体接口设计开始，按你当前仓库结构一步步实现。
