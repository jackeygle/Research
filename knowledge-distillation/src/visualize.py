"""
Generate visualization plots for Knowledge Distillation experiment results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory
os.makedirs('results/figures', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# =============================================================================
# Data from experiments
# =============================================================================

# Main results
models = ['Teacher\n(ResNet-34)', 'Student\nBaseline', 'Student\n+ KD']
accuracies = [94.08, 92.10, 92.95]
params = [21.28, 2.24, 2.24]  # in millions
colors = ['#2ecc71', '#e74c3c', '#3498db']

# Temperature ablation (alpha=0.3)
temperatures = [1, 2, 4, 8, 16]
temp_acc = [92.64, 92.51, 92.58, 92.72, 92.95]

# Alpha ablation (T=4)
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
alpha_acc = [92.19, 92.58, 92.89, 92.58, 92.66]

# =============================================================================
# Figure 1: Main Results Comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
ax1 = axes[0]
bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylim([90, 95])
ax1.axhline(y=92.10, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Add value labels
for bar, acc in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# Parameters comparison
ax2 = axes[1]
bars2 = ax2.bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Parameters (Millions)')
ax2.set_title('Model Size Comparison')

# Add value labels
for bar, p in zip(bars2, params):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{p:.2f}M', ha='center', va='bottom', fontweight='bold')

# Add compression ratio annotation
ax2.annotate('~10x\nCompression!', xy=(1.5, 10), fontsize=14, 
             ha='center', color='#e74c3c', fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/main_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figures/main_results.png")

# =============================================================================
# Figure 2: Ablation Studies
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Temperature ablation
ax1 = axes[0]
ax1.plot(temperatures, temp_acc, 'o-', color='#3498db', linewidth=2, markersize=10)
ax1.axhline(y=92.10, color='gray', linestyle='--', alpha=0.7, label='Baseline (no KD)')
ax1.set_xlabel('Temperature (T)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Temperature Ablation (α=0.3)')
ax1.set_xticks(temperatures)
ax1.set_ylim([91.5, 93.5])
ax1.legend()

# Highlight best
best_idx = np.argmax(temp_acc)
ax1.scatter([temperatures[best_idx]], [temp_acc[best_idx]], 
            color='#e74c3c', s=200, zorder=5, marker='*')
ax1.annotate(f'Best: {temp_acc[best_idx]:.2f}%', 
             xy=(temperatures[best_idx], temp_acc[best_idx]),
             xytext=(temperatures[best_idx]-3, temp_acc[best_idx]+0.3),
             fontsize=11, fontweight='bold', color='#e74c3c')

# Alpha ablation
ax2 = axes[1]
ax2.plot(alphas, alpha_acc, 'o-', color='#9b59b6', linewidth=2, markersize=10)
ax2.axhline(y=92.10, color='gray', linestyle='--', alpha=0.7, label='Baseline (no KD)')
ax2.set_xlabel('Alpha (α)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Alpha Ablation (T=4)')
ax2.set_xticks(alphas)
ax2.set_ylim([91.5, 93.5])
ax2.legend()

# Highlight best
best_idx = np.argmax(alpha_acc)
ax2.scatter([alphas[best_idx]], [alpha_acc[best_idx]], 
            color='#e74c3c', s=200, zorder=5, marker='*')
ax2.annotate(f'Best: {alpha_acc[best_idx]:.2f}%', 
             xy=(alphas[best_idx], alpha_acc[best_idx]),
             xytext=(alphas[best_idx]+0.1, alpha_acc[best_idx]+0.2),
             fontsize=11, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig('results/figures/ablation_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figures/ablation_study.png")

# =============================================================================
# Figure 3: Knowledge Distillation Concept
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Create conceptual diagram data
x = np.linspace(0, 9, 10)
hard_labels = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # One-hot for class 3
soft_labels = np.array([0.01, 0.02, 0.15, 0.50, 0.12, 0.08, 0.05, 0.03, 0.02, 0.02])

width = 0.35
ax.bar(x - width/2, hard_labels, width, label='Hard Labels', color='#3498db', alpha=0.8)
ax.bar(x + width/2, soft_labels, width, label='Soft Labels (Teacher)', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Class')
ax.set_ylabel('Probability')
ax.set_title('Hard Labels vs Soft Labels (Dark Knowledge)')
ax.set_xticks(x)
ax.set_xticklabels(['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 
                   rotation=45, ha='right')
ax.legend()

# Add annotation
ax.annotate('Dark Knowledge:\nSoft labels reveal\nclass relationships!', 
            xy=(5, 0.12), xytext=(6.5, 0.35),
            fontsize=12, ha='center',
            arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/figures/kd_concept.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figures/kd_concept.png")

# =============================================================================
# Figure 4: Summary Table as Image
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Table data
table_data = [
    ['Model', 'Accuracy', 'Parameters', 'Compression', 'Improvement'],
    ['Teacher (ResNet-34)', '94.08%', '21.28M', '1x', '-'],
    ['Student Baseline', '92.10%', '2.24M', '~10x', 'Baseline'],
    ['Student + KD (T=4, α=0.3)', '92.58%', '2.24M', '~10x', '+0.48%'],
    ['Student + KD (T=16, α=0.3)', '92.95%', '2.24M', '~10x', '+0.85%'],
    ['Student + KD (T=4, α=0.5)', '92.89%', '2.24M', '~10x', '+0.79%'],
]

colors_table = [['#34495e']*5, 
                ['#27ae60']*5,
                ['#e74c3c']*5, 
                ['#3498db']*5, 
                ['#2ecc71']*5,
                ['#3498db']*5]

table = ax.table(cellText=table_data, 
                 cellLoc='center',
                 loc='center',
                 cellColours=[['#ecf0f1']*5] + [['white']*5]*5)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# Highlight best result
table[(4, 1)].set_facecolor('#d5f5e3')
table[(4, 4)].set_facecolor('#d5f5e3')

plt.title('Knowledge Distillation Results Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig('results/figures/summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figures/summary_table.png")

print("\n✅ All figures generated successfully!")
print("Figures saved in: results/figures/")
