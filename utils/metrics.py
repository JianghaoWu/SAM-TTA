import torch
import torch.nn.functional as F


def calculate_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """Compute Dice coefficient between binary predicted and ground-truth masks."""
    smooth = 1e-6
    intersection = (pred_mask * gt_mask).sum()
    return (2. * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)


def calculate_nsd(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Compute Normalized Surface Dice (NSD) between predicted and ground-truth masks."""
    pred_mask = (pred_mask > threshold).float()
    gt_mask = (gt_mask > threshold).float()

    pred_edges = torch.clamp(pred_mask - F.max_pool2d(pred_mask.unsqueeze(0), 3, stride=1, padding=1).squeeze(0), min=0)
    gt_edges = torch.clamp(gt_mask - F.max_pool2d(gt_mask.unsqueeze(0), 3, stride=1, padding=1).squeeze(0), min=0)

    intersection = (pred_edges * gt_edges).sum()
    union = (pred_edges + gt_edges).sum()
    return intersection / (union + 1e-6)
