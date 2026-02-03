"""
Evaluation Script for Video Action Recognition
"""
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dataset import UCF101Dataset, get_dataloaders
from model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Action Recognition Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate model on dataset.
    
    Returns:
        all_preds: All predictions
        all_labels: All ground truth labels
        all_probs: All prediction probabilities
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for videos, labels in tqdm(dataloader, desc="Evaluating"):
        videos = videos.to(device, non_blocking=True)
        
        outputs = model(videos)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str,
    top_k: int = 20
):
    """Plot confusion matrix for top-k most frequent classes."""
    # Get top-k most frequent classes
    unique, counts = np.unique(y_true, return_counts=True)
    top_classes = unique[np.argsort(counts)[-top_k:]]
    
    # Filter to top classes
    mask = np.isin(y_true, top_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Map to consecutive indices
    class_mapping = {c: i for i, c in enumerate(top_classes)}
    y_true_mapped = np.array([class_mapping[y] for y in y_true_filtered])
    y_pred_mapped = np.array([class_mapping.get(y, -1) for y in y_pred_filtered])
    
    # Only keep valid predictions
    valid_mask = y_pred_mapped >= 0
    y_true_mapped = y_true_mapped[valid_mask]
    y_pred_mapped = y_pred_mapped[valid_mask]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot
    plt.figure(figsize=(16, 14))
    labels = [class_names[c] for c in top_classes]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Top {top_k} Classes)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def plot_accuracy_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: str,
    top_k: int = 30
):
    """Plot accuracy per class."""
    # Calculate per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    
    accuracies = []
    for cls in range(len(class_names)):
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls] * 100
        else:
            acc = 0
        accuracies.append((cls, acc, class_total[cls]))
    
    # Sort by accuracy
    accuracies_sorted = sorted(accuracies, key=lambda x: x[1], reverse=True)
    
    # Plot top and bottom classes
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Top classes
    top_accs = accuracies_sorted[:top_k]
    ax = axes[0]
    bars = ax.barh(
        [class_names[c[0]][:20] for c in top_accs],
        [c[1] for c in top_accs],
        color='green',
        alpha=0.7
    )
    ax.set_xlabel('Accuracy (%)')
    ax.set_title(f'Top {top_k} Classes by Accuracy')
    ax.invert_yaxis()
    
    # Bottom classes
    bottom_accs = accuracies_sorted[-top_k:]
    ax = axes[1]
    bars = ax.barh(
        [class_names[c[0]][:20] for c in bottom_accs],
        [c[1] for c in bottom_accs],
        color='red',
        alpha=0.7
    )
    ax.set_xlabel('Accuracy (%)')
    ax.set_title(f'Bottom {top_k} Classes by Accuracy')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved per-class accuracy: {output_path}")


def measure_inference_speed(
    model: nn.Module,
    device: torch.device,
    num_frames: int = 16,
    num_runs: int = 100
) -> dict:
    """Measure inference speed."""
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(1, 3, num_frames, 112, 112).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    import time
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    
    times = np.array(times) * 1000  # ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'fps': 1000 / np.mean(times)
    }


def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model_config = config.get('model', {})
    model = create_model(
        model_type=model_config.get('type', 'r3d_18'),
        num_classes=model_config.get('num_classes', 101),
        pretrained=False,
        dropout=model_config.get('dropout', 0.5)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    data_config = config.get('data', {})
    data_dir = args.data_dir or data_config.get('data_dir', '/scratch/work/zhangx29/data/UCF101')
    
    _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        num_frames=data_config.get('num_frames', 16),
        frame_size=tuple(data_config.get('frame_size', [112, 112]))
    )
    
    class_names = test_loader.dataset.classes
    print(f"Number of classes: {len(class_names)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\nEvaluating model...")
    preds, labels, probs = evaluate(model, test_loader, device)
    
    # Calculate metrics
    top1_acc = (preds == labels).mean() * 100
    
    # Top-5 accuracy
    top5_correct = 0
    for i, (label, prob) in enumerate(zip(labels, probs)):
        top5_preds = prob.argsort()[-5:]
        if label in top5_preds:
            top5_correct += 1
    top5_acc = top5_correct / len(labels) * 100
    
    print(f"\nResults:")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save classification report
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    report_path = os.path.join(args.output_dir, 'classification_report.yaml')
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False)
    print(f"Saved classification report: {report_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels, preds, class_names, cm_path)
    
    # Plot per-class accuracy
    acc_path = os.path.join(args.output_dir, 'per_class_accuracy.png')
    plot_accuracy_per_class(labels, preds, class_names, acc_path)
    
    # Measure inference speed
    print("\nMeasuring inference speed...")
    speed = measure_inference_speed(model, device)
    print(f"  Inference time: {speed['mean_ms']:.2f} ± {speed['std_ms']:.2f} ms")
    print(f"  FPS: {speed['fps']:.1f}")
    
    # Save summary
    summary = {
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'num_classes': len(class_names),
        'num_test_samples': len(labels),
        'inference_time_ms': float(speed['mean_ms']),
        'fps': float(speed['fps'])
    }
    
    summary_path = os.path.join(args.output_dir, 'evaluation_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    print(f"\nSaved evaluation summary: {summary_path}")


if __name__ == '__main__':
    main()
