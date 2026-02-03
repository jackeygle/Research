"""
Utility functions for data loading, training helpers, and logging.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import yaml


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed_mode() -> tuple[bool, int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return True, rank, world_size, local_rank, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, test_loader
    """
    return get_dataloader('cifar10', data_dir, batch_size, num_workers)


def _resolve_normalization(dataset_key: str, normalize: str) -> Tuple[list, list]:
    normalize = (normalize or "auto").lower()
    if normalize == "auto":
        if dataset_key in {"cifar10", "cifar100"}:
            normalize = "cifar"
        elif dataset_key == "tinyimagenet":
            normalize = "imagenet"
    if normalize == "cifar":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif normalize == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        raise ValueError(f"Unknown normalize option: {normalize}")
    return mean, std


def _build_transforms(
    base_size: int,
    image_size: Optional[int],
    mean: list,
    std: list,
    train: bool
) -> transforms.Compose:
    if image_size is None or image_size == base_size:
        if train:
            return transforms.Compose([
                transforms.RandomCrop(base_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def get_dataloader(
    dataset: str = 'cifar10',
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: Optional[int] = None,
    normalize: str = "auto",
    distributed: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test data loaders for CIFAR-10, CIFAR-100, or Tiny-ImageNet.
    
    Args:
        dataset: 'cifar10', 'cifar100', or 'tinyimagenet'
        data_dir: Directory to store/load dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Optional input resolution override (e.g., 224 for ViT)
        normalize: 'auto' | 'cifar' | 'imagenet' | 'clip'
    
    Returns:
        train_loader, test_loader
    """
    dataset_key = dataset.lower().replace("-", "").replace("_", "")

    if dataset_key in {'cifar10', 'cifar100'}:
        mean, std = _resolve_normalization(dataset_key, normalize)
        base_size = 32
        train_transform = _build_transforms(base_size, image_size, mean, std, train=True)
        test_transform = _build_transforms(base_size, image_size, mean, std, train=False)

        # Select dataset
        if dataset_key == 'cifar10':
            DatasetClass = torchvision.datasets.CIFAR10
        else:
            DatasetClass = torchvision.datasets.CIFAR100

        train_dataset = DatasetClass(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = DatasetClass(
            root=data_dir, train=False, download=True, transform=test_transform
        )
    elif dataset_key == 'tinyimagenet':
        # Tiny-ImageNet (200 classes, 64x64)
        mean, std = _resolve_normalization(dataset_key, normalize)
        base_size = 64
        train_transform = _build_transforms(base_size, image_size, mean, std, train=True)
        test_transform = _build_transforms(base_size, image_size, mean, std, train=False)

        tiny_root = os.path.join(data_dir, 'tiny-imagenet-200')
        train_dir = os.path.join(tiny_root, 'train')
        val_dir = os.path.join(tiny_root, 'val')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                "Tiny-ImageNet not found. Run scripts/prepare_tinyimagenet.sh "
                "or place the dataset at data/tiny-imagenet-200."
            )

        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(val_dir, transform=test_transform)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Use 'cifar10', 'cifar100', or 'tinyimagenet'"
        )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        sampler=test_sampler, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def get_num_classes(dataset: str) -> int:
    """Get number of classes for a dataset."""
    dataset_key = dataset.lower().replace("-", "").replace("_", "")
    if dataset_key == 'cifar10':
        return 10
    elif dataset_key == 'cifar100':
        return 100
    elif dataset_key == 'tinyimagenet':
        return 200
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    path: str
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }, path)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler."""
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


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

    def all_reduce(self):
        if not is_dist_avail_and_initialized():
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.sum = t[0].item()
        self.count = t[1].item()
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """
    Get a module from model by its name (e.g., 'features.0').
    
    Args:
        model: PyTorch model
        name: Module name, can be dot-separated (e.g., 'features.0')
    
    Returns:
        The requested module
    """
    names = name.split('.')
    module = model
    for n in names:
        if hasattr(module, n):
            module = getattr(module, n)
        elif hasattr(module, '__getitem__'):
            module = module[int(n)]
        else:
            raise AttributeError(f"Module {model.__class__.__name__} has no attribute '{n}'")
    return module


class FeatureHook:
    """
    Hook to capture intermediate feature maps from a model.
    Used for feature-based distillation methods (FitNets, Attention Transfer).
    """
    
    def __init__(self, module: nn.Module):
        self.module = module
        self.features = None
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Store the output feature map."""
        self.features = output
    
    def remove(self):
        """Remove the hook."""
        self.hook.remove()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


if __name__ == "__main__":
    # Test data loading
    print("Testing CIFAR-10 data loading...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
