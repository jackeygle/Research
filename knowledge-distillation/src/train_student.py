"""
Train student model with or without Knowledge Distillation.

Usage:
    # Train with KD
    python src/train_student.py --teacher-ckpt checkpoints/teacher_best.pth
    
    # Train without KD (baseline)
    python src/train_student.py --no-kd
"""

import argparse
import copy
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models import get_teacher, get_student, count_parameters, load_checkpoint
from distillation import (
    DistillationLoss, StudentLoss, FeatureDistillationLoss,
    AttentionTransferLoss, ContrastiveDistillationLoss
)
from utils import (
    set_seed, get_dataloader, get_num_classes, save_checkpoint,
    get_lr_scheduler, AverageMeter, accuracy, get_module_by_name, FeatureHook,
    init_distributed_mode, is_main_process, unwrap_model
)


def _match_spatial_size(student_feat, teacher_feat):
    if student_feat.dim() != 4 or teacher_feat.dim() != 4:
        return student_feat, teacher_feat
    target_h = min(student_feat.size(2), teacher_feat.size(2))
    target_w = min(student_feat.size(3), teacher_feat.size(3))
    if (student_feat.size(2), student_feat.size(3)) != (target_h, target_w):
        student_feat = F.adaptive_avg_pool2d(student_feat, (target_h, target_w))
    if (teacher_feat.size(2), teacher_feat.size(3)) != (target_h, target_w):
        teacher_feat = F.adaptive_avg_pool2d(teacher_feat, (target_h, target_w))
    return student_feat, teacher_feat


def update_ema_teacher(ema_teacher, student, decay):
    with torch.no_grad():
        for ema_param, student_param in zip(ema_teacher.parameters(), student.parameters()):
            ema_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)


def train_epoch_distill(
    student,
    teacher,
    train_loader,
    hard_criterion,
    distill_criterion,
    optimizer,
    device,
    distill_method,
    alpha,
    hooks=None,
    ema_teacher=None,
    ema_decay=0.999,
    max_batches=None,
    distributed=False,
    is_main=True
):
    """Train student with selected distillation method for one epoch."""
    student.train()
    if teacher is not None:
        teacher.eval()
    if ema_teacher is not None:
        ema_teacher.eval()
    
    losses = AverageMeter()
    hard_losses = AverageMeter()
    distill_losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Training ({distill_method.upper()})', disable=not is_main)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        if distill_method in {'kd', 'self'}:
            with torch.no_grad():
                teacher_logits = ema_teacher(images) if distill_method == 'self' else teacher(images)
            student_logits = student(images)
            loss, loss_dict = distill_criterion(student_logits, teacher_logits, labels)
            hard_loss = loss_dict['hard_loss']
            distill_loss = loss_dict['soft_loss']
        else:
            with torch.no_grad():
                _ = teacher(images)
                teacher_feat = hooks['teacher'].features
            student_logits = student(images)
            student_feat = hooks['student'].features
            student_feat, teacher_feat = _match_spatial_size(student_feat, teacher_feat)

            hard_loss = hard_criterion(student_logits, labels)
            distill_loss = distill_criterion(student_feat, teacher_feat)
            loss = alpha * hard_loss + (1 - alpha) * distill_loss

            loss_dict = {
                'total_loss': loss.item(),
                'hard_loss': hard_loss.item(),
                'soft_loss': distill_loss.item()
            }

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if distill_method == 'self' and ema_teacher is not None:
            update_ema_teacher(ema_teacher, student, decay=ema_decay)
        
        # Metrics
        acc = accuracy(student_logits, labels)[0]
        losses.update(loss_dict['total_loss'], images.size(0))
        hard_losses.update(loss_dict['hard_loss'], images.size(0))
        distill_losses.update(loss_dict['soft_loss'], images.size(0))
        top1.update(acc, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{top1.avg:.2f}%'
        })
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    
    if distributed:
        losses.all_reduce()
        hard_losses.all_reduce()
        distill_losses.all_reduce()
        top1.all_reduce()
    return {
        'loss': losses.avg,
        'hard_loss': hard_losses.avg,
        'soft_loss': distill_losses.avg,
        'accuracy': top1.avg
    }


def train_epoch_dml(
    student_a,
    student_b,
    train_loader,
    distill_criterion,
    optimizer_a,
    optimizer_b,
    device,
    max_batches=None,
    distributed=False,
    is_main=True
):
    """Train two students with Deep Mutual Learning (DML) for one epoch."""
    student_a.train()
    student_b.train()

    losses_a = AverageMeter()
    losses_b = AverageMeter()
    hard_losses_a = AverageMeter()
    hard_losses_b = AverageMeter()
    distill_losses_a = AverageMeter()
    distill_losses_b = AverageMeter()
    top1_a = AverageMeter()
    top1_b = AverageMeter()

    pbar = tqdm(train_loader, desc='Training (DML)', disable=not is_main)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        logits_a = student_a(images)
        logits_b = student_b(images)

        loss_a, loss_dict_a = distill_criterion(logits_a, logits_b.detach(), labels)
        loss_b, loss_dict_b = distill_criterion(logits_b, logits_a.detach(), labels)
        total_loss = loss_a + loss_b

        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        total_loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        acc_a = accuracy(logits_a, labels)[0]
        acc_b = accuracy(logits_b, labels)[0]

        losses_a.update(loss_dict_a['total_loss'], images.size(0))
        losses_b.update(loss_dict_b['total_loss'], images.size(0))
        hard_losses_a.update(loss_dict_a['hard_loss'], images.size(0))
        hard_losses_b.update(loss_dict_b['hard_loss'], images.size(0))
        distill_losses_a.update(loss_dict_a['soft_loss'], images.size(0))
        distill_losses_b.update(loss_dict_b['soft_loss'], images.size(0))
        top1_a.update(acc_a, images.size(0))
        top1_b.update(acc_b, images.size(0))

        pbar.set_postfix({
            'loss_a': f'{losses_a.avg:.4f}',
            'acc_a': f'{top1_a.avg:.2f}%',
            'loss_b': f'{losses_b.avg:.4f}',
            'acc_b': f'{top1_b.avg:.2f}%'
        })
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break

    if distributed:
        losses_a.all_reduce()
        losses_b.all_reduce()
        hard_losses_a.all_reduce()
        hard_losses_b.all_reduce()
        distill_losses_a.all_reduce()
        distill_losses_b.all_reduce()
        top1_a.all_reduce()
        top1_b.all_reduce()
    return {
        'loss_a': losses_a.avg,
        'loss_b': losses_b.avg,
        'hard_loss_a': hard_losses_a.avg,
        'hard_loss_b': hard_losses_b.avg,
        'soft_loss_a': distill_losses_a.avg,
        'soft_loss_b': distill_losses_b.avg,
        'accuracy_a': top1_a.avg,
        'accuracy_b': top1_b.avg
    }


def train_epoch_baseline(
    student,
    train_loader,
    criterion,
    optimizer,
    device,
    max_batches=None,
    distributed=False,
    is_main=True
):
    """Train student without distillation for one epoch."""
    student.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc='Training (Baseline)', disable=not is_main)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = student(images)
        loss, loss_dict = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = accuracy(outputs, labels)[0]
        losses.update(loss_dict['total_loss'], images.size(0))
        top1.update(acc, images.size(0))
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}', 'acc': f'{top1.avg:.2f}%'})
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    
    if distributed:
        losses.all_reduce()
        top1.all_reduce()
    return {'loss': losses.avg, 'accuracy': top1.avg}


def validate(model, test_loader, device, max_batches=None, distributed=False, is_main=True):
    """Validate model on test set."""
    model.eval()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc='Validating', disable=not is_main)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            acc = accuracy(outputs, labels)[0]
            top1.update(acc, images.size(0))
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    
    if distributed:
        top1.all_reduce()
    return top1.avg


def main(args):
    # Setup
    distributed, rank, world_size, local_rank, device = init_distributed_mode()
    set_seed(args.seed + rank)
    is_main = is_main_process()
    if is_main:
        print(f"Using device: {device}")
    ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if distributed and device.type == "cuda" else {}
    
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

    input_size = args.image_size
    if input_size is None:
        input_size = 64 if args.dataset.lower() == 'tinyimagenet' else 32

    is_dml = (not args.no_kd and args.distill_method == 'dml')

    # Student model(s)
    if is_dml:
        peer_arch = args.peer_arch or args.student_arch
        peer_width_mult = args.peer_width_mult if args.peer_width_mult is not None else args.width_mult

        student_a = get_student(
            num_classes=num_classes, width_mult=args.width_mult, arch=args.student_arch,
            input_size=input_size
        )
        student_b = get_student(
            num_classes=num_classes, width_mult=peer_width_mult, arch=peer_arch,
            input_size=input_size
        )
        student_a = student_a.to(device)
        student_b = student_b.to(device)
        if distributed:
            student_a = DDP(student_a, **ddp_kwargs)
            student_b = DDP(student_b, **ddp_kwargs)
        if is_main:
            print(f"Student A ({args.student_arch}): {count_parameters(unwrap_model(student_a)):,} parameters")
            print(f"Student B ({peer_arch}): {count_parameters(unwrap_model(student_b)):,} parameters")

        teacher = None
        ema_teacher = None
        hard_criterion = None
        distill_criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
        exp_name = (
            f'dml_{args.student_arch}_{peer_arch}_{args.dataset}_'
            f'T{args.temperature}_a{args.alpha}'
        )
    else:
        student = get_student(
            num_classes=num_classes, width_mult=args.width_mult, arch=args.student_arch,
            input_size=input_size
        )
        student = student.to(device)
        if distributed:
            student = DDP(student, **ddp_kwargs)
        if is_main:
            print(f"Student model ({args.student_arch}): {count_parameters(unwrap_model(student)):,} parameters")

        # Teacher model (if using KD)
        teacher = None
        ema_teacher = None
        if not args.no_kd and args.distill_method not in {'self'}:
            teacher = get_teacher(
                num_classes=num_classes, pretrained=False, arch=args.teacher_arch,
                input_size=input_size
            )
            teacher = load_checkpoint(teacher, args.teacher_ckpt)
            teacher = teacher.to(device)
            teacher.eval()
            if is_main:
                print(f"Teacher model ({args.teacher_arch}) loaded from: {args.teacher_ckpt}")
                print(f"Teacher parameters: {count_parameters(teacher):,}")

        # Loss function and experiment name
        if args.no_kd:
            hard_criterion = None
            distill_criterion = StudentLoss()
            exp_name = f'student_{args.student_arch}_baseline_{args.dataset}'
        else:
            hard_criterion = nn.CrossEntropyLoss()
            if args.distill_method in {'kd', 'self'}:
                distill_criterion = DistillationLoss(temperature=args.temperature, alpha=args.alpha)
            elif args.distill_method == 'fitnets':
                distill_criterion = None
            elif args.distill_method == 'attention':
                distill_criterion = AttentionTransferLoss(weight=args.feature_weight)
            elif args.distill_method == 'contrastive':
                distill_criterion = None
            else:
                raise ValueError(f"Unknown distill method: {args.distill_method}")

            exp_name = (
                f'student_{args.student_arch}_{args.distill_method}_{args.dataset}_'
                f'T{args.temperature}_a{args.alpha}'
            )

    # Feature hooks for feature-based distillation
    hooks = None
    if not args.no_kd and not is_dml and args.distill_method in {'fitnets', 'attention', 'contrastive'}:
        if args.teacher_layer is None:
            if args.teacher_arch.startswith('vit'):
                args.teacher_layer = 'encoder.ln'
            else:
                args.teacher_layer = 'layer4'
        if args.student_layer is None:
            if args.student_arch in {'resnet18', 'wide_resnet50_2'}:
                args.student_layer = 'layer4'
            elif args.student_arch.startswith('mobilenet'):
                args.student_layer = 'features.18'
            elif args.student_arch.startswith('efficientnet'):
                args.student_layer = 'features.7'
            else:
                args.student_layer = 'layer4'

        teacher_module = get_module_by_name(teacher, args.teacher_layer)
        student_module = get_module_by_name(unwrap_model(student), args.student_layer)
        hooks = {
            'teacher': FeatureHook(teacher_module),
            'student': FeatureHook(student_module)
        }

        # Warmup to infer channel dims for projector
        with torch.no_grad():
            images, _ = next(iter(train_loader))
            images = images.to(device)
            _ = teacher(images)
            _ = student(images)
        teacher_feat = hooks['teacher'].features
        student_feat = hooks['student'].features
        if teacher_feat is None or student_feat is None:
            raise RuntimeError("Feature hooks did not capture outputs. Check layer names.")

        if args.distill_method == 'fitnets':
            distill_criterion = FeatureDistillationLoss(
                student_channels=student_feat.size(1),
                teacher_channels=teacher_feat.size(1),
                weight=args.feature_weight
            )
        elif args.distill_method == 'contrastive':
            def _feat_dim(feat: torch.Tensor) -> int:
                return feat.size(1) if feat.dim() == 4 else feat.size(-1)

            student_dim = _feat_dim(student_feat)
            teacher_dim = _feat_dim(teacher_feat)
            distill_criterion = ContrastiveDistillationLoss(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                temperature=args.contrastive_temp,
                weight=args.feature_weight
            )

    # Self-distillation uses EMA teacher
    if not args.no_kd and not is_dml and args.distill_method == 'self':
        ema_teacher = copy.deepcopy(unwrap_model(student)).to(device)
        for param in ema_teacher.parameters():
            param.requires_grad = False

    # Optimizer
    if is_dml:
        distill_criterion = distill_criterion.to(device)
        optimizer_a = optim.SGD(
            student_a.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        optimizer_b = optim.SGD(
            student_b.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler_a = get_lr_scheduler(optimizer_a, args.scheduler, args.epochs)
        scheduler_b = get_lr_scheduler(optimizer_b, args.scheduler, args.epochs)
    else:
        optimizer_params = list(student.parameters())
        if distill_criterion is not None:
            distill_criterion = distill_criterion.to(device)
            extra_params = [p for p in distill_criterion.parameters() if p.requires_grad]
            if distributed and extra_params:
                distill_criterion = DDP(distill_criterion, **ddp_kwargs)
                extra_params = [p for p in distill_criterion.parameters() if p.requires_grad]
            if extra_params:
                optimizer_params += extra_params
        optimizer = optim.SGD(
            optimizer_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = get_lr_scheduler(optimizer, args.scheduler, args.epochs)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.log_dir, exp_name)) if is_main else None
    
    # Training loop
    if is_dml:
        best_acc_a = 0.0
        best_acc_b = 0.0
        best_acc_avg = 0.0
        for epoch in range(1, args.epochs + 1):
            if distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if is_main:
                print(f"\nEpoch {epoch}/{args.epochs}")
                print(
                    f"Learning rate A/B: {optimizer_a.param_groups[0]['lr']:.6f} "
                    f"/ {optimizer_b.param_groups[0]['lr']:.6f}"
                )

            train_metrics = train_epoch_dml(
                student_a, student_b, train_loader, distill_criterion,
                optimizer_a, optimizer_b, device,
                max_batches=args.max_train_batches,
                distributed=distributed,
                is_main=is_main
            )

            val_acc_a = validate(
                student_a, test_loader, device,
                max_batches=args.max_val_batches,
                distributed=distributed,
                is_main=is_main
            )
            val_acc_b = validate(
                student_b, test_loader, device,
                max_batches=args.max_val_batches,
                distributed=distributed,
                is_main=is_main
            )
            val_acc_avg = (val_acc_a + val_acc_b) / 2.0

            scheduler_a.step()
            scheduler_b.step()

            if writer is not None:
                writer.add_scalar('Loss/train_a', train_metrics['loss_a'], epoch)
                writer.add_scalar('Loss/train_b', train_metrics['loss_b'], epoch)
                writer.add_scalar('Loss/hard_a', train_metrics['hard_loss_a'], epoch)
                writer.add_scalar('Loss/hard_b', train_metrics['hard_loss_b'], epoch)
                writer.add_scalar('Loss/soft_a', train_metrics['soft_loss_a'], epoch)
                writer.add_scalar('Loss/soft_b', train_metrics['soft_loss_b'], epoch)
                writer.add_scalar('Accuracy/train_a', train_metrics['accuracy_a'], epoch)
                writer.add_scalar('Accuracy/train_b', train_metrics['accuracy_b'], epoch)
                writer.add_scalar('Accuracy/val_a', val_acc_a, epoch)
                writer.add_scalar('Accuracy/val_b', val_acc_b, epoch)
                writer.add_scalar('Accuracy/val_avg', val_acc_avg, epoch)
                writer.add_scalar('LR/a', optimizer_a.param_groups[0]['lr'], epoch)
                writer.add_scalar('LR/b', optimizer_b.param_groups[0]['lr'], epoch)

            if is_main:
                print(
                    f"Train Acc A/B: {train_metrics['accuracy_a']:.2f}% / "
                    f"{train_metrics['accuracy_b']:.2f}%"
                )
                print(f"Val Acc A/B/Avg: {val_acc_a:.2f}% / {val_acc_b:.2f}% / {val_acc_avg:.2f}%")

            if val_acc_a > best_acc_a:
                best_acc_a = val_acc_a
                if is_main:
                    save_checkpoint(
                        unwrap_model(student_a), optimizer_a, epoch, val_acc_a,
                        os.path.join(args.checkpoint_dir, f'{exp_name}_A_best.pth')
                    )
                    print(f"New best A accuracy: {best_acc_a:.2f}%")

            if val_acc_b > best_acc_b:
                best_acc_b = val_acc_b
                if is_main:
                    save_checkpoint(
                        unwrap_model(student_b), optimizer_b, epoch, val_acc_b,
                        os.path.join(args.checkpoint_dir, f'{exp_name}_B_best.pth')
                    )
                    print(f"New best B accuracy: {best_acc_b:.2f}%")

            if val_acc_avg > best_acc_avg:
                best_acc_avg = val_acc_avg

        if writer is not None:
            writer.close()
        if is_main:
            print(
                f"\nTraining complete! Best A/B/Avg accuracy: "
                f"{best_acc_a:.2f}% / {best_acc_b:.2f}% / {best_acc_avg:.2f}%"
            )

        results = {
            'experiment': exp_name,
            'best_accuracy': best_acc_avg,
            'best_accuracy_a': best_acc_a,
            'best_accuracy_b': best_acc_b,
            'parameters_a': count_parameters(unwrap_model(student_a)),
            'parameters_b': count_parameters(unwrap_model(student_b)),
            'dataset': args.dataset,
            'use_kd': True,
            'distill_method': 'dml',
            'temperature': args.temperature,
            'alpha': args.alpha,
            'student_arch': args.student_arch,
            'peer_arch': args.peer_arch or args.student_arch,
            'width_mult': args.width_mult,
            'peer_width_mult': args.peer_width_mult if args.peer_width_mult is not None else args.width_mult,
        }

        results_file = os.path.join(args.checkpoint_dir, f'{exp_name}_results.txt')
        if is_main:
            with open(results_file, 'w') as f:
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")
    else:
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            if distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if is_main:
                print(f"\nEpoch {epoch}/{args.epochs}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Train
            if args.no_kd:
                train_metrics = train_epoch_baseline(
                    student, train_loader, distill_criterion, optimizer, device,
                    max_batches=args.max_train_batches,
                    distributed=distributed,
                    is_main=is_main
                )
            else:
                train_metrics = train_epoch_distill(
                    student, teacher, train_loader, hard_criterion, distill_criterion,
                    optimizer, device, args.distill_method, args.alpha,
                    hooks=hooks, ema_teacher=ema_teacher, ema_decay=args.ema_decay,
                    max_batches=args.max_train_batches,
                    distributed=distributed,
                    is_main=is_main
                )

            # Validate
            val_acc = validate(
                student, test_loader, device,
                max_batches=args.max_val_batches,
                distributed=distributed,
                is_main=is_main
            )

            # Step scheduler
            scheduler.step()

            # Log
            if writer is not None:
                writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            if not args.no_kd:
                if writer is not None:
                    writer.add_scalar('Loss/hard', train_metrics['hard_loss'], epoch)
                    writer.add_scalar('Loss/soft', train_metrics['soft_loss'], epoch)

            if is_main:
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"Val Acc: {val_acc:.2f}%")

            # Save best checkpoint
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                if is_main:
                    save_checkpoint(
                        unwrap_model(student), optimizer, epoch, val_acc,
                        os.path.join(args.checkpoint_dir, f'{exp_name}_best.pth')
                    )
                    print(f"New best accuracy: {best_acc:.2f}%")

        if writer is not None:
            writer.close()
        if is_main:
            print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")

        # Save final results
        results = {
            'experiment': exp_name,
            'best_accuracy': best_acc,
            'parameters': count_parameters(unwrap_model(student)),
            'dataset': args.dataset,
            'use_kd': not args.no_kd,
            'distill_method': None if args.no_kd else args.distill_method,
            'temperature': args.temperature if not args.no_kd else None,
            'alpha': args.alpha if not args.no_kd else None,
            'teacher_layer': args.teacher_layer if hooks is not None else None,
            'student_layer': args.student_layer if hooks is not None else None,
            'feature_weight': args.feature_weight if not args.no_kd else None,
            'contrastive_temp': args.contrastive_temp if args.distill_method == 'contrastive' else None,
        }

        results_file = os.path.join(args.checkpoint_dir, f'{exp_name}_results.txt')
        if is_main:
            with open(results_file, 'w') as f:
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Student Model')
    
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tinyimagenet'], help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model
    parser.add_argument('--student-arch', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                                 'efficientnet_b0', 'resnet18', 'wide_resnet50_2'],
                        help='Student architecture')
    parser.add_argument('--width-mult', type=float, default=1.0,
                        help='Width multiplier for MobileNet variants')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Override input resolution (e.g., 224 for ViT)')
    parser.add_argument('--normalize', type=str, default='auto',
                        choices=['auto', 'cifar', 'imagenet', 'clip'],
                        help='Normalization stats to use')
    parser.add_argument('--peer-arch', type=str, default=None,
                        choices=['mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
                                 'efficientnet_b0', 'resnet18', 'wide_resnet50_2'],
                        help='Peer architecture for DML (defaults to student-arch)')
    parser.add_argument('--peer-width-mult', type=float, default=None,
                        help='Width multiplier for DML peer (defaults to width-mult)')
    
    # Teacher (for KD)
    parser.add_argument('--teacher-arch', type=str, default='resnet34',
                        choices=['resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'wide_resnet50_2', 'efficientnet_b4', 'vit_b_16',
                                 'vit_l_16', 'clip_vit_b32'],
                        help='Teacher architecture (must match checkpoint)')
    parser.add_argument('--teacher-ckpt', type=str, default='./checkpoints/teacher_best.pth')
    parser.add_argument('--no-kd', action='store_true', help='Train without KD (baseline)')
    
    # Distillation
    parser.add_argument('--distill-method', type=str, default='kd',
                        choices=['kd', 'fitnets', 'attention', 'self', 'contrastive', 'dml'])
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--feature-weight', type=float, default=1.0)
    parser.add_argument('--contrastive-temp', type=float, default=0.1)
    parser.add_argument('--teacher-layer', type=str, default=None,
                        help='Feature hook for teacher (auto-selected if omitted)')
    parser.add_argument('--student-layer', type=str, default=None,
                        help='Feature hook for student (auto-selected if omitted)')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine')
    
    # Logging
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-batches', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    
    args = parser.parse_args()
    main(args)
