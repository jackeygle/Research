#!/bin/bash
# Quick test script to verify code works
# Run this locally before submitting SLURM jobs

# Load Python environment
module load mamba
source activate kd

cd /scratch/work/zhangx29/knowledge-distillation

echo "Testing Python modules..."

# Test models
echo "1. Testing models.py..."
python -c "
from src.models import get_teacher, get_student, count_parameters
import torch

teacher = get_teacher(num_classes=10)
student = get_student(num_classes=10)

print(f'  Teacher: {count_parameters(teacher):,} params')
print(f'  Student: {count_parameters(student):,} params')

x = torch.randn(2, 3, 32, 32)
print(f'  Teacher output: {teacher(x).shape}')
print(f'  Student output: {student(x).shape}')
print('  ✓ models.py OK')
"

# Test distillation loss
echo "2. Testing distillation.py..."
python -c "
from src.distillation import DistillationLoss
import torch

criterion = DistillationLoss(temperature=4.0, alpha=0.3)
student_logits = torch.randn(8, 10)
teacher_logits = torch.randn(8, 10)
labels = torch.randint(0, 10, (8,))

loss, loss_dict = criterion(student_logits, teacher_logits, labels)
print(f'  Loss: {loss.item():.4f}')
print('  ✓ distillation.py OK')
"

# Test data loading
echo "3. Testing utils.py (data loading)..."
python -c "
from src.utils import get_cifar10_loaders
train_loader, test_loader = get_cifar10_loaders(batch_size=32, num_workers=0)
print(f'  Train batches: {len(train_loader)}')
print(f'  Test batches: {len(test_loader)}')
print('  ✓ utils.py OK')
"

echo ""
echo "All tests passed! Ready to submit SLURM jobs."
echo ""
echo "Next steps:"
echo "  1. sbatch scripts/train_teacher.sh"
echo "  2. Wait for teacher to finish"
echo "  3. sbatch scripts/train_student.sh --no-kd"
echo "  4. sbatch scripts/train_student.sh"
