"""
R3D-18 Model for Video Action Recognition
"""
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from typing import Optional


class ActionRecognitionModel(nn.Module):
    """
    R3D-18 based action recognition model.
    
    Uses pretrained weights from Kinetics-400 and fine-tunes on target dataset.
    
    Args:
        num_classes: Number of action classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate before final classifier
    """
    
    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Load R3D-18 backbone
        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)
        
        # Get feature dimension
        in_features = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B = batch size
               C = 3 (RGB channels)
               T = number of frames (typically 16)
               H, W = height, width (typically 112x112)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier.
        
        Useful for visualization and transfer learning.
        """
        # Forward through all layers except fc
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.flatten(1)
        return x
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone layers for fine-tuning."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = not freeze
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class EfficientActionModel(nn.Module):
    """
    A more efficient variant using (2+1)D convolutions.
    
    (2+1)D factorizes 3D convolutions into spatial and temporal components,
    reducing parameters while maintaining performance.
    """
    
    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        
        if pretrained:
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            self.backbone = r2plus1d_18(weights=weights)
        else:
            self.backbone = r2plus1d_18(weights=None)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(
    model_type: str = 'r3d_18',
    num_classes: int = 101,
    pretrained: bool = True,
    dropout: float = 0.5
) -> nn.Module:
    """
    Factory function to create action recognition model.
    
    Args:
        model_type: 'r3d_18' or 'r2plus1d_18'
        num_classes: Number of classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    if model_type == 'r3d_18':
        return ActionRecognitionModel(num_classes, pretrained, dropout)
    elif model_type == 'r2plus1d_18':
        return EfficientActionModel(num_classes, pretrained, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model
    model = create_model('r3d_18', num_classes=101, pretrained=True)
    print(f"Model: R3D-18")
    print(f"Total parameters: {model.get_num_params(trainable_only=False):,}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 16, 112, 112)  # (B, C, T, H, W)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test feature extraction
    with torch.no_grad():
        features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
