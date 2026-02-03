# Knowledge Distillation for Neural Network Compression

A PyTorch implementation of Knowledge Distillation on CIFAR-10, demonstrating how smaller models can learn from larger models.

## Project Structure
```
knowledge-distillation/
├── src/
│   ├── models.py           # Teacher & Student models
│   ├── distillation.py     # KD loss
│   ├── train_teacher.py    # Train teacher
│   ├── train_student.py    # Train student with/without KD
│   ├── prune_utils.py      # Unstructured & structured pruning
│   ├── prune_student.py    # Distill-then-Prune pipeline
│   └── evaluate.py         # Evaluation
├── configs/
│   ├── config.yaml
│   └── pruning.yaml        # Pruning config
├── scripts/                # SLURM job scripts
├── docs/                   # Plans (e.g. PRUNING_PLAN.md)
└── results/                # Experiment outputs
```

## Quick Start

```bash
# Train teacher model
sbatch scripts/train_teacher.sh

# Train student without KD (baseline)
sbatch scripts/train_student.sh --no-kd

# Train student with KD
sbatch scripts/train_student.sh
```

## Distill-then-Prune

Prune a KD-trained student (structured channel pruning or L1 unstructured). Install optional deps: `pip install torch-pruning thop`.

```bash
# Prune student (structured 30%, finetune 5 epochs). Uses checkpoint from env or default.
DATASET=cifar10 PRUNE_RATIO=0.3 FINETUNE_EPOCHS=5 bash scripts/run_prune_student.sh

# Or run directly:
python src/prune_student.py --student-ckpt ./checkpoints/student_kd_cifar10_T4.0_a0.3_w1.0_best.pth \
  --dataset cifar10 --prune-method structured --prune-ratio 0.3 --finetune-epochs 5
```

Results are written to `checkpoints/*_pruned_*_results.txt` and `checkpoints/pruning_comparison.csv`.

## Tiny-ImageNet (200 classes, 64x64)

```bash
# Download and prepare Tiny-ImageNet
bash scripts/prepare_tinyimagenet.sh

# Train teacher on Tiny-ImageNet
python src/train_teacher.py --dataset tinyimagenet --data-dir ./data

# Train student with KD on Tiny-ImageNet
python src/train_student.py --dataset tinyimagenet --data-dir ./data --teacher-ckpt ./checkpoints/teacher_tinyimagenet_best.pth
```

## Results

| Model | Accuracy | Parameters | Compression |
|-------|----------|------------|-------------|
| Teacher (ResNet-34) | ~94% | 21M | 1.0x |
| Student baseline | ~85% | 3.5M | 6x |
| Student + KD | ~90% | 3.5M | 6x |

## Knowledge Distillation

The core idea: train a small "student" model to mimic a large "teacher" model's soft predictions.

```
L_total = α × L_CE(student, labels) + (1-α) × T² × L_KL(soft_student, soft_teacher)
```

- **Temperature (T)**: Softens probability distributions
- **Alpha (α)**: Balance between hard and soft labels

## Author
Xinle Zhang - Aalto University
