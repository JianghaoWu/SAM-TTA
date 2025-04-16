import torch

def calculate_dice(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    计算 Dice 系数.

    Parameters:
    pred_mask (torch.Tensor): 预测的二值掩膜
    gt_mask (torch.Tensor): 真实的二值掩膜

    Returns:
    torch.Tensor: 计算得到的 Dice 系数
    """
    smooth = 1e-6  # 平滑因子，避免除零错误

    intersection = (pred_mask * gt_mask).sum()
    dice = (2. * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)
    return dice

import torch
import torch.nn.functional as F

def calculate_nsd(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    计算 Normalized Surface Dice (NSD).
    
    Parameters:
    pred_mask (torch.Tensor): 预测的二值掩膜
    gt_mask (torch.Tensor): 真实的二值掩膜
    threshold (float): 二值化的阈值，用于控制表面计算的敏感度

    Returns:
    torch.Tensor: 计算得到的 Normalized Surface Dice (NSD)
    """
    # 使用 threshold 二值化掩膜，确保掩膜为0或1
    pred_mask = (pred_mask > threshold).float()
    gt_mask = (gt_mask > threshold).float()

    # 计算边缘：使用膨胀和腐蚀来找到边缘
    pred_edges = pred_mask - F.max_pool2d(pred_mask.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    gt_edges = gt_mask - F.max_pool2d(gt_mask.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)

    # 去除边缘值中的负数（可能是由于边缘计算中的问题）
    pred_edges = torch.clamp(pred_edges, min=0)
    gt_edges = torch.clamp(gt_edges, min=0)

    # 计算边缘的交集与并集
    intersection = (pred_edges * gt_edges).sum()
    union = (pred_edges + gt_edges).sum()

    # 计算 NSD
    nsd = intersection / (union + 1e-6)  # 加一个小的平滑因子避免除零
    return nsd
