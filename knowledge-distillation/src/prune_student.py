"""
Distill-then-Prune: load trained student -> prune -> optional fine-tune -> evaluate -> save.

Usage:
    # Prune KD-trained student (structured 30%), then finetune 5 epochs
    python src/prune_student.py --student-ckpt checkpoints/student_kd_cifar10_*_best.pth \\
        --dataset cifar10 --prune-method structured --prune-ratio 0.3 --finetune-epochs 5

    # Unstructured 30%, no finetune
    python src/prune_student.py --student-ckpt checkpoints/student_kd_cifar10_*_best.pth \\
        --dataset cifar10 --prune-method unstructured --prune-ratio 0.3
"""

from __future__ import annotations

import argparse
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from models import get_student, load_checkpoint, count_parameters
from utils import (
    set_seed,
    get_dataloader,
    get_num_classes,
    accuracy,
    AverageMeter,
    save_checkpoint,
)
from prune_utils import (
    apply_unstructured_prune,
    apply_structured_prune,
    get_student_ignored_layers,
    count_flops_params,
    get_model_size_mb,
)


def _inference_time_ms(model: nn.Module, device: torch.device, input_shape: tuple) -> float:
    dummy = torch.randn(1, *input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / 100 * 1000


def _evaluate(model: nn.Module, test_loader, device: torch.device) -> dict:
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            a1, a5 = accuracy(out, labels, topk=(1, 5))
            top1.update(a1, images.size(0))
            top5.update(a5, images.size(0))
    return {"top1": top1.avg, "top5": top5.avg}


def _finetune(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    num_epochs: int,
    lr: float,
) -> float:
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for ep in range(1, num_epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Finetune {ep}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
        res = _evaluate(model, test_loader, device)
        if res["top1"] > best_acc:
            best_acc = res["top1"]
        print(f"  Finetune epoch {ep}: val acc {res['top1']:.2f}%")
    return best_acc


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    num_classes = get_num_classes(args.dataset)
    input_hw = 64 if args.dataset.lower() == "tinyimagenet" else 32
    input_shape = (3, input_hw, input_hw)

    train_loader, test_loader = get_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    student = get_student(num_classes=num_classes, width_mult=args.width_mult)
    load_checkpoint(student, args.student_ckpt)
    student = student.to(device)

    example_input = torch.randn(1, *input_shape).to(device)
    params_before = count_parameters(student)
    size_before = get_model_size_mb(student)
    flops_before, _ = count_flops_params(student, example_input)
    res_before = _evaluate(student, test_loader, device)
    time_before = _inference_time_ms(student, device, input_shape)

    print(f"\nBefore pruning: params={params_before:,} size={size_before:.2f} MB "
          f"FLOPs={flops_before/1e6:.2f}M top1={res_before['top1']:.2f}% time={time_before:.2f} ms")

    if args.prune_method == "unstructured":
        npruned = apply_unstructured_prune(student, amount=args.prune_ratio)
        print(f"Applied L1 unstructured pruning: {args.prune_ratio*100:.0f}% -> {npruned:,} weights zeroed")
        is_structured = False
    else:
        ignored = get_student_ignored_layers(student)
        dparams, dmacs = apply_structured_prune(
            student,
            example_input,
            amount=args.prune_ratio,
            ignored_layers=ignored,
            iterative_steps=args.iterative_steps,
        )
        print(f"Applied structured channel pruning: {args.prune_ratio*100:.0f}% -> Δparams={dparams:,} ΔMACs={dmacs:,}")
        is_structured = True

    params_after = count_parameters(student)
    size_after = get_model_size_mb(student)
    flops_after, _ = count_flops_params(student, example_input)
    res_after = _evaluate(student, test_loader, device)
    time_after = _inference_time_ms(student, device, input_shape)

    print(f"After pruning: params={params_after:,} size={size_after:.2f} MB "
          f"FLOPs={flops_after/1e6:.2f}M top1={res_after['top1']:.2f}% time={time_after:.2f} ms")

    if args.finetune_epochs > 0:
        print(f"\nFinetuning for {args.finetune_epochs} epochs (lr={args.finetune_lr})...")
        best_acc = _finetune(
            student, train_loader, test_loader, device,
            num_epochs=args.finetune_epochs,
            lr=args.finetune_lr,
        )
        res_after = _evaluate(student, test_loader, device)
        time_after = _inference_time_ms(student, device, input_shape)
        print(f"After finetune: top1={res_after['top1']:.2f}% time={time_after:.2f} ms")

    base = os.path.splitext(os.path.basename(args.student_ckpt))[0]
    if base.endswith("_best"):
        base = base[:-5]
    exp_name = f"{base}_pruned_{args.prune_method}_{int(args.prune_ratio*100)}pct"

    if is_structured:
        out_path = os.path.join(args.checkpoint_dir, f"{exp_name}.pth")
        torch.save({
            "model": student.cpu().eval(),
            "pruned_structured": True,
            "dataset": args.dataset,
            "prune_ratio": args.prune_ratio,
            "prune_method": args.prune_method,
            "accuracy": res_after["top1"],
        }, out_path)
        student = student.to(device)
        print(f"Saved structured pruned model (full) -> {out_path}")
    else:
        out_path = os.path.join(args.checkpoint_dir, f"{exp_name}_best.pth")
        save_checkpoint(student, optim.SGD(student.parameters(), lr=1e-3), 0, res_after["top1"], out_path)
        print(f"Saved unstructured pruned checkpoint -> {out_path}")

    results = {
        "experiment": exp_name,
        "student_ckpt": args.student_ckpt,
        "dataset": args.dataset,
        "prune_method": args.prune_method,
        "prune_ratio": args.prune_ratio,
        "finetune_epochs": args.finetune_epochs,
        "params_before": params_before,
        "params_after": params_after,
        "size_mb_before": round(size_before, 2),
        "size_mb_after": round(size_after, 2),
        "flops_before": flops_before,
        "flops_after": flops_after,
        "top1_before": round(res_before["top1"], 2),
        "top1_after": round(res_after["top1"], 2),
        "top5_after": round(res_after["top5"], 2),
        "inference_ms_after": round(time_after, 2),
    }
    results_file = os.path.join(args.checkpoint_dir, f"{exp_name}_results.txt")
    with open(results_file, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"Results -> {results_file}")

    csv_path = os.path.join(args.checkpoint_dir, "pruning_comparison.csv")
    write_pruning_csv_row(results, csv_path)
    print(f"Appended to {csv_path}")

    return results


def write_pruning_csv_row(results: dict, csv_path: str):
    row = (
        f"{results['experiment']},"
        f"{results['prune_method']},{results['prune_ratio']},"
        f"{results['params_before']},{results['params_after']},"
        f"{results['size_mb_before']},{results['size_mb_after']},"
        f"{results['top1_before']},{results['top1_after']},"
        f"{results['inference_ms_after']}\n"
    )
    if not os.path.isfile(csv_path):
        header = "experiment,prune_method,prune_ratio,params_before,params_after,size_mb_before,size_mb_after,top1_before,top1_after,inference_ms_after\n"
        with open(csv_path, "w") as f:
            f.write(header)
    with open(csv_path, "a") as f:
        f.write(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Distill-then-Prune: prune trained student")
    ap.add_argument("--student-ckpt", type=str, required=True, help="Path to trained student checkpoint")
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "tinyimagenet"])
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--width-mult", type=float, default=1.0)

    ap.add_argument("--prune-method", type=str, default="structured",
                    choices=["unstructured", "structured"])
    ap.add_argument("--prune-ratio", type=float, default=0.3,
                    help="Fraction of params/channels to remove (e.g. 0.3 = 30%%)")
    ap.add_argument("--iterative-steps", type=int, default=1,
                    help="Steps for structured pruning (1 = one-shot)")

    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--finetune-lr", type=float, default=1e-3)

    ap.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
