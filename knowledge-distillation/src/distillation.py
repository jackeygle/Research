"""
Knowledge Distillation loss implementation.

The total loss is:
L = α * L_CE(student, labels) + (1-α) * T² * L_KL(student_soft, teacher_soft)

where:
- L_CE: Cross-entropy loss with hard labels
- L_KL: KL divergence between softened predictions
- T: Temperature (higher = softer distributions)
- α: Balance between hard and soft labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    
    Combines hard label cross-entropy with soft label KL divergence.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3):
        """
        Args:
            temperature: Softening temperature for logits
            alpha: Weight for hard label loss (1-alpha for soft label loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Raw logits from student model [B, C]
            teacher_logits: Raw logits from teacher model [B, C]
            labels: Ground truth labels [B]
        
        Returns:
            total_loss: Combined distillation loss
            loss_dict: Dictionary with individual loss components
        """
        # Hard label loss (cross-entropy with ground truth)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft label loss (KL divergence with teacher's soft predictions)
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence, scaled by T^2 as per Hinton et al.
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
        }
        
        return total_loss, loss_dict


class FeatureDistillationLoss(nn.Module):
    """
    FitNets-style feature distillation with a 1x1 projection.

    Aligns student features to teacher features using MSE loss.
    """

    def __init__(self, student_channels: int, teacher_channels: int, weight: float = 1.0):
        super().__init__()
        self.projector = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)
        self.weight = weight

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        teacher_feat = teacher_feat.detach()
        student_proj = self.projector(student_feat)
        return self.weight * F.mse_loss(student_proj, teacher_feat)


class AttentionTransferLoss(nn.Module):
    """
    Attention transfer loss.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        teacher_feat = teacher_feat.detach()
        s_attn = self._attention_map(student_feat)
        t_attn = self._attention_map(teacher_feat)
        return self.weight * F.mse_loss(s_attn, t_attn)

    @staticmethod
    def _attention_map(feat: torch.Tensor) -> torch.Tensor:
        # Sum of squared feature maps across channels
        attn = feat.pow(2).mean(dim=1, keepdim=True)
        # Normalize spatially
        attn = attn.view(attn.size(0), -1)
        attn = F.normalize(attn, p=2, dim=1)
        return attn


class ContrastiveDistillationLoss(nn.Module):
    """
    Contrastive distillation loss (InfoNCE) on global pooled features.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        temperature: float = 0.1,
        weight: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.projector = None

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        teacher_feat = teacher_feat.detach()
        s = self._pool_and_normalize(student_feat)
        t = self._pool_and_normalize(teacher_feat)

        if self.projector is not None:
            s = self.projector(s)
            s = F.normalize(s, p=2, dim=1)

        logits = torch.mm(s, t.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return self.weight * F.cross_entropy(logits, labels)

    @staticmethod
    def _pool_and_normalize(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        elif feat.dim() == 3:
            # Handle token sequences (e.g., ViT encoder output) by using CLS token.
            feat = feat[:, 0] if feat.size(1) > 1 else feat.mean(dim=1)
        return F.normalize(feat, p=2, dim=1)


class StudentLoss(nn.Module):
    """Standard cross-entropy loss for training student without distillation."""
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        loss = self.ce_loss(logits, labels)
        return loss, {'total_loss': loss.item(), 'hard_loss': loss.item()}


if __name__ == "__main__":
    # Test distillation loss
    batch_size = 32
    num_classes = 10
    
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test different temperatures
    for temp in [1.0, 4.0, 10.0, 20.0]:
        criterion = DistillationLoss(temperature=temp, alpha=0.3)
        loss, loss_dict = criterion(student_logits, teacher_logits, labels)
        print(f"T={temp:>4.1f}: total={loss_dict['total_loss']:.4f}, "
              f"hard={loss_dict['hard_loss']:.4f}, soft={loss_dict['soft_loss']:.4f}")