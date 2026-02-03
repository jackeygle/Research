"""
UCF101 Dataset for Video Action Recognition
"""
import os
import random
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.io import read_video
import numpy as np


class UCF101Dataset(Dataset):
    """
    UCF101 Dataset for video classification.
    
    Args:
        root_dir: Path to UCF101 dataset
        split: 'train' or 'test'
        num_frames: Number of frames to sample per video
        frame_size: Target frame size (H, W)
        transform: Optional additional transforms
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (112, 112),
        transform: Optional[Callable] = None,
        split_file: Optional[str] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        
        # Load class names
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load video paths
        self.samples = self._load_samples(split_file)
        
        # Default transforms
        self.normalize = T.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989]
        )
        
    def _load_samples(self, split_file: Optional[str] = None) -> List[Tuple[str, int]]:
        """Load video paths and labels."""
        samples = []
        
        if split_file and os.path.exists(split_file):
            # Load from official split file
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        video_path = parts[0]
                        class_name = video_path.split('/')[0]
                        if class_name in self.class_to_idx:
                            full_path = self.root_dir / video_path
                            if full_path.exists():
                                samples.append((str(full_path), self.class_to_idx[class_name]))
        else:
            # Load all videos from directory structure
            for class_name in self.classes:
                class_dir = self.root_dir / class_name
                for video_file in class_dir.glob('*.avi'):
                    samples.append((str(video_file), self.class_to_idx[class_name]))
            
            # Simple train/test split (80/20)
            random.seed(42)
            random.shuffle(samples)
            split_idx = int(len(samples) * 0.8)
            
            if self.split == 'train':
                samples = samples[:split_idx]
            else:
                samples = samples[split_idx:]
        
        return samples
    
    def _sample_frames(self, video_path: str) -> torch.Tensor:
        """
        Sample frames uniformly from video.
        
        Returns:
            Tensor of shape (C, T, H, W)
        """
        try:
            # Read video
            video, _, info = read_video(video_path, pts_unit='sec')
            # video shape: (T, H, W, C)
            
            total_frames = video.shape[0]
            
            if total_frames == 0:
                raise ValueError(f"Empty video: {video_path}")
            
            # Uniform sampling
            if total_frames >= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # Repeat frames if video is too short
                indices = np.arange(total_frames)
                indices = np.tile(indices, (self.num_frames // total_frames) + 1)[:self.num_frames]
            
            frames = video[indices]  # (T, H, W, C)
            
            # Convert to (C, T, H, W) for 3D CNN
            frames = frames.permute(3, 0, 1, 2).float() / 255.0
            
            # Resize
            frames = torch.nn.functional.interpolate(
                frames.unsqueeze(0),
                size=(self.num_frames, self.frame_size[0], self.frame_size[1]),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return zeros on error
            return torch.zeros(3, self.num_frames, self.frame_size[0], self.frame_size[1])
    
    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation for training."""
        if self.split == 'train':
            # Random horizontal flip
            if random.random() > 0.5:
                frames = torch.flip(frames, dims=[-1])
            
            # Random crop (with padding)
            if random.random() > 0.5:
                crop_size = int(self.frame_size[0] * 0.9)
                i = random.randint(0, self.frame_size[0] - crop_size)
                j = random.randint(0, self.frame_size[1] - crop_size)
                frames = frames[:, :, i:i+crop_size, j:j+crop_size]
                frames = torch.nn.functional.interpolate(
                    frames.unsqueeze(0),
                    size=(self.num_frames, self.frame_size[0], self.frame_size[1]),
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Color jitter (brightness, contrast)
            if random.random() > 0.5:
                brightness = 1.0 + random.uniform(-0.2, 0.2)
                frames = frames * brightness
                frames = torch.clamp(frames, 0, 1)
        
        return frames
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        
        # Sample frames
        frames = self._sample_frames(video_path)
        
        # Augment
        frames = self._augment(frames)
        
        # Normalize
        # Reshape for normalization: (C, T, H, W) -> (T, C, H, W) -> normalize -> (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W)
        frames = torch.stack([self.normalize(f) for f in frames])  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        # Additional transform if provided
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (112, 112)
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get train and test dataloaders.
    """
    train_dataset = UCF101Dataset(
        root_dir=data_dir,
        split='train',
        num_frames=num_frames,
        frame_size=frame_size
    )
    
    test_dataset = UCF101Dataset(
        root_dir=data_dir,
        split='test',
        num_frames=num_frames,
        frame_size=frame_size
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    dataset = UCF101Dataset(
        root_dir='/scratch/work/zhangx29/data/UCF101',
        split='train',
        num_frames=16
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    if len(dataset) > 0:
        sample, label = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Label: {label} ({dataset.classes[label]})")
