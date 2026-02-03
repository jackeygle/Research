"""
Training Script for Video Action Recognition
"""
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import UCF101Dataset, get_dataloaders
from model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Action Recognition Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    default_config = {
        'data': {
            'data_dir': '/scratch/work/zhangx29/data/UCF101',
            'num_frames': 16,
            'frame_size': [112, 112],
            'num_workers': 4
        },
        'model': {
            'type': 'r3d_18',
            'num_classes': 101,
            'pretrained': True,
            'dropout': 0.5
        },
        'training': {
            'epochs': 30,
            'batch_size': 8,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'lr_scheduler': 'cosine',
            'warmup_epochs': 3,
            'use_amp': True
        },
        'output': {
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'results/logs',
            'save_every': 5
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
            elif isinstance(default_config[key], dict):
                for subkey in default_config[key]:
                    if subkey not in config[key]:
                        config[key][subkey] = default_config[key][subkey]
        return config
    
    return default_config


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True
) -> dict:
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = model(videos)
            loss = criterion(outputs, labels)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # Update meters
        batch_size = videos.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} '
                  f'Top-1: {top1.avg:.2f}% '
                  f'Top-5: {top5.avg:.2f}% '
                  f'Time: {batch_time.avg:.3f}s')
    
    return {
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for videos, labels in val_loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        batch_size = videos.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
    
    return {
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg
    }


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Compute top-k accuracy."""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


def get_scheduler(optimizer, config, num_batches):
    """Create learning rate scheduler."""
    scheduler_type = config['training'].get('lr_scheduler', 'cosine')
    epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    
    if scheduler_type == 'cosine':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(epochs - warmup_epochs) * num_batches
        )
    elif scheduler_type == 'step':
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10 * num_batches,
            gamma=0.1
        )
    else:
        main_scheduler = None
    
    return main_scheduler


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str
):
    """Save checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Saved checkpoint: {filepath}")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    
    # Print config
    print("=" * 50)
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        num_frames=config['data']['num_frames'],
        frame_size=tuple(config['data']['frame_size'])
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=config['model']['type'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # TensorBoard
    log_dir = os.path.join(
        config['output']['log_dir'],
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)
    
    # Training loop
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, config['training']['use_amp']
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Acc/train_top1', train_metrics['top1'], epoch)
        writer.add_scalar('Acc/val_top1', val_metrics['top1'], epoch)
        writer.add_scalar('Acc/train_top5', train_metrics['top5'], epoch)
        writer.add_scalar('Acc/val_top5', val_metrics['top5'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Top-1: {train_metrics['top1']:.2f}%, Top-5: {train_metrics['top5']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Top-1: {val_metrics['top1']:.2f}%, Top-5: {val_metrics['top5']:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['top1'] > best_acc
        best_acc = max(val_metrics['top1'], best_acc)
        
        if is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                },
                config['output']['checkpoint_dir'],
                'best_model.pth'
            )
        
        if (epoch + 1) % config['output']['save_every'] == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                },
                config['output']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch}.pth'
            )
    
    print("\n" + "=" * 50)
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print("=" * 50)
    
    writer.close()


if __name__ == '__main__':
    main()
