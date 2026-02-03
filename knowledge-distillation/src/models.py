"""
Models for Knowledge Distillation experiments.
Teacher: ResNet family (34/50/101/152) or EfficientNet
Student: MobileNetV2/V3 or EfficientNet-B0
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


# Supported teacher architectures
TEACHER_ARCHS = [
    'resnet34', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2',
    'efficientnet_b4', 'vit_b_16', 'vit_l_16', 'clip_vit_b32'
]
# Supported student architectures
STUDENT_ARCHS = [
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
    'efficientnet_b0', 'resnet18', 'wide_resnet50_2'
]


class ClipImageTeacher(nn.Module):
    """CLIP image encoder with a classification head."""

    def __init__(self, clip_model: nn.Module, num_classes: int, freeze_visual: bool = False):
        super().__init__()
        self.clip_model = clip_model
        if freeze_visual:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        # LazyLinear avoids depending on a specific CLIP embed dim.
        self.head = nn.LazyLinear(num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.clip_model.encode_image(images)
        if feats.dim() > 2:
            feats = feats.flatten(1)
        return self.head(feats)


def get_teacher(
    num_classes: int = 10,
    pretrained: bool = True,
    arch: str = 'resnet34',
    input_size: Optional[int] = None
) -> nn.Module:
    """
    Get teacher model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        arch: Architecture name ('resnet34', 'resnet50', 'resnet101', 'resnet152', 
              'wide_resnet50_2', 'efficientnet_b4')
    
    Returns:
        Teacher model adapted for CIFAR/TinyImageNet
    """
    arch = arch.lower()
    
    if arch == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    elif arch == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    elif arch == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet101(weights=weights)
    elif arch == 'resnet152':
        weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet152(weights=weights)
    elif arch == 'wide_resnet50_2':
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.wide_resnet50_2(weights=weights)
    elif arch == 'efficientnet_b4':
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        # EfficientNet has different structure
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif arch == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    elif arch == 'vit_l_16':
        weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_l_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    elif arch == 'clip_vit_b32':
        try:
            import open_clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for clip_vit_b32. "
                "Install with: pip install open_clip_torch"
            ) from exc
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        model = ClipImageTeacher(clip_model, num_classes=num_classes)
        return model
    else:
        raise ValueError(f"Unknown teacher architecture: {arch}. Supported: {TEACHER_ARCHS}")
    
    # Adapt ResNet family for CIFAR/TinyImageNet (smaller images)
    if input_size is None or input_size <= 64:
        # Replace first conv layer: 7x7 -> 3x3, stride 2 -> 1
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool for small images
    
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def get_student(
    num_classes: int = 10, 
    width_mult: float = 1.0,
    arch: str = 'mobilenet_v2',
    input_size: Optional[int] = None
) -> nn.Module:
    """
    Get student model.
    
    Args:
        num_classes: Number of output classes
        width_mult: Width multiplier (for MobileNet variants)
        arch: Architecture name ('mobilenet_v2', 'mobilenet_v3_small',
              'mobilenet_v3_large', 'efficientnet_b0', 'resnet18', 'wide_resnet50_2')
    
    Returns:
        Student model adapted for CIFAR/TinyImageNet
    """
    arch = arch.lower()
    
    is_small_input = input_size is None or input_size <= 64

    if arch == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None, width_mult=width_mult)
        # Adapt for smaller images
        if is_small_input:
            model.features[0][0] = nn.Conv2d(
                3, int(32 * width_mult), kernel_size=3, stride=1, padding=1, bias=False
            )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
    elif arch == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=None)
        # Adapt for smaller images
        if is_small_input:
            model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    elif arch == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        # Adapt for smaller images
        if is_small_input:
            model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif arch == 'resnet18':
        model = models.resnet18(weights=None)
        # Adapt for smaller images
        if is_small_input:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights=None)
        if is_small_input:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown student architecture: {arch}. Supported: {STUDENT_ARCHS}")
    
    return model


def get_student_small(num_classes: int = 10) -> nn.Module:
    """
    Get a smaller student model (MobileNetV2 with width_mult=0.5).
    Even more compression, good for ablation studies.
    """
    return get_student(num_classes=num_classes, width_mult=0.5, arch='mobilenet_v2')


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


if __name__ == "__main__":
    num_classes = 200  # TinyImageNet
    
    print("=" * 60)
    print("TEACHER MODELS")
    print("=" * 60)
    for arch in TEACHER_ARCHS:
        try:
            model = get_teacher(num_classes=num_classes, pretrained=False, arch=arch)
            params = count_parameters(model)
            print(f"{arch:20s}: {params:>12,} parameters ({params/1e6:.1f}M)")
        except Exception as e:
            print(f"{arch:20s}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("STUDENT MODELS")
    print("=" * 60)
    for arch in STUDENT_ARCHS:
        try:
            model = get_student(num_classes=num_classes, arch=arch)
            params = count_parameters(model)
            print(f"{arch:20s}: {params:>12,} parameters ({params/1e6:.1f}M)")
        except Exception as e:
            print(f"{arch:20s}: Error - {e}")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("FORWARD PASS TEST (32x32 input)")
    print("=" * 60)
    x = torch.randn(2, 3, 32, 32)
    teacher = get_teacher(num_classes=10, arch='resnet50')
    student = get_student(num_classes=10, arch='mobilenet_v3_large')
    print(f"ResNet-50 output shape: {teacher(x).shape}")
    print(f"MobileNetV3-Large output shape: {student(x).shape}")
