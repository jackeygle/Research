"""
Train teacher model (ResNet-34) on CIFAR-10.

Usage:
    python src/train_teacher.py --epochs 100 --lr 0.1
"""

import argparse
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models import get_teacher, count_parameters
from utils import (
    set_seed, get_dataloader, get_num_classes, save_checkpoint,
    get_lr_scheduler, AverageMeter, accuracy, init_distributed_mode,
    is_main_process, unwrap_model
)


def train_epoch(model, train_loader, criterion, optimizer, device, distributed=False, is_main=True):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training', disable=not is_main)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = accuracy(outputs, labels)[0]
        losses.update(loss.item(), images.size(0))
        top1.update(acc, images.size(0))
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}%'})
    
    if distributed:
        losses.all_reduce()
        top1.all_reduce()
    return losses.avg, top1.avg


def validate(model, test_loader, criterion, device, distributed=False, is_main=True):
    """Validate model on test set."""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Validating', disable=not is_main):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc = accuracy(outputs, labels)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))
    
    if distributed:
        losses.all_reduce()
        top1.all_reduce()
    return losses.avg, top1.avg


def main(args):
    # Setup
    distributed, rank, world_size, local_rank, device = init_distributed_mode()
    set_seed(args.seed + rank)
    is_main = is_main_process()
    if is_main:
        print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Data
    num_classes = get_num_classes(args.dataset)
    train_loader, test_loader = get_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        normalize=args.normalize,
        distributed=distributed
    )
    if is_main:
        print(f"Dataset: {args.dataset.upper()} ({num_classes} classes)")
    
    # Model
    input_size = args.image_size
    if input_size is None:
        input_size = 64 if args.dataset.lower() == 'tinyimagenet' else 32
    model = get_teacher(
        num_classes=num_classes,
        pretrained=args.pretrained,
        arch=args.arch,
        input_size=input_size
    )
    model = model.to(device)
    if distributed:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    if is_main:
        print(f"Teacher model ({args.arch}): {count_parameters(unwrap_model(model)):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            if is_main:
                print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint.get('accuracy', 0.0)
            # Adjust scheduler to correct position
            for _ in range(start_epoch - 1):
                scheduler.step()
            if is_main:
                print(f"Resumed from epoch {checkpoint['epoch']}, best acc: {best_acc:.2f}%")
        else:
            if is_main:
                print(f"Warning: checkpoint not found: {args.resume}")
    
    # Tensorboard
    exp_name = f'teacher_{args.arch}_{args.dataset}'
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name)) if is_main else None
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if is_main:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            distributed=distributed, is_main=is_main
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device,
            distributed=distributed, is_main=is_main
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        if is_main:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            if is_main:
                save_checkpoint(
                    unwrap_model(model), optimizer, epoch, val_acc,
                    os.path.join(args.checkpoint_dir, f'teacher_{args.arch}_{args.dataset}_best.pth')
                )
                print(f"New best accuracy: {best_acc:.2f}%")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            if is_main:
                save_checkpoint(
                    unwrap_model(model), optimizer, epoch, val_acc,
                    os.path.join(args.checkpoint_dir, f'teacher_{args.arch}_{args.dataset}_epoch{epoch}.pth')
                )
    
    if writer is not None:
        writer.close()
    if is_main:
        print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet'], help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model
    parser.add_argument('--arch', type=str, default='resnet34',
                        choices=['resnet34', 'resnet50', 'resnet101', 'resnet152', 
                                 'wide_resnet50_2', 'efficientnet_b4', 'vit_b_16',
                                 'vit_l_16', 'clip_vit_b32'],
                        help='Teacher architecture')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--image-size', type=int, default=None,
                        help='Override input resolution (e.g., 224 for ViT)')
    parser.add_argument('--normalize', type=str, default='auto',
                        choices=['auto', 'cifar', 'imagenet', 'clip'],
                        help='Normalization stats to use')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    # Logging
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--save-freq', type=int, default=20)
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
