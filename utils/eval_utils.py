import os
import torch
import lightning as L
import segmentation_models_pytorch as smp
from box import Box
from torch.utils.data import DataLoader
from model import Model
from utils.sample_utils import get_point_prompts
from utils.tools import write_csv
from medpy import metric
import numpy as np
from typing import Tuple
from PIL import Image
import pandas as pd


class AverageMeter:
    """Computes and stores the running average and current value."""

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


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    batch_iou = intersection / (union + 1e-7)
    return batch_iou.unsqueeze(1)


def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse" or 'box' in cfg.prompt:
        return bboxes
    elif cfg.prompt == "point":
        return get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError(f"Unknown prompt type: {cfg.prompt}")


def calculate_metrics(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Compute segmentation metrics between predicted and ground-truth masks.

    Returns:
        (Dice, ASSD, HD95, IoU) — all capped at 50.0 for surface distance metrics.
        Returns (1.0, 0.0, 0.0, 1.0) when both masks are empty.
        Returns (0.0, 50.0, 50.0, 0.0) when only one mask is empty.
    """
    pred_mask_np = pred_mask.cpu().numpy()
    gt_mask_np = gt_mask.cpu().numpy()

    if np.sum(pred_mask_np) == 0 and np.sum(gt_mask_np) == 0:
        return 1.0, 0.0, 0.0, 1.0

    if np.sum(pred_mask_np) > 0 and np.sum(gt_mask_np) > 0:
        batch_dice = metric.binary.dc(pred_mask_np, gt_mask_np)
        batch_asd = min(metric.binary.asd(pred_mask_np, gt_mask_np), 50.0)
        batch_hd95 = min(metric.binary.hd95(pred_mask_np, gt_mask_np), 50.0)
        intersection = np.logical_and(pred_mask_np, gt_mask_np).sum()
        union = np.logical_or(pred_mask_np, gt_mask_np).sum()
        batch_iou = float(intersection) / float(union + 1e-7)
        return batch_dice, batch_asd, batch_hd95, batch_iou

    # One mask is empty, the other is not
    return metric.binary.dc(pred_mask_np, gt_mask_np), 50.0, 50.0, 0.0


def test_med(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    model.eval()
    dice_scores = AverageMeter()
    assd_scores = AverageMeter()
    hd95_scores = AverageMeter()
    case_results = []

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, basenames = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)
            _, pred_masks, _, _ = model(images, prompts)

            for pred_mask, gt_mask, basename in zip(pred_masks, gt_masks, basenames):
                pred_mask = (pred_mask > 0.5).float()
                gt_mask = (gt_mask > 0.5).float()

                batch_dice, batch_assd, batch_hd95, _ = calculate_metrics(pred_mask, gt_mask)
                dice_scores.update(batch_dice, num_images)
                assd_scores.update(batch_assd, num_images)
                hd95_scores.update(batch_hd95, num_images)

                pred_mask_np = (pred_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
                save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basename}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(pred_mask_np).save(save_path)

                case_results.append({
                    "basename": basename,
                    "Dice": batch_dice,
                    "ASSD": batch_assd,
                    "HD95": batch_hd95,
                })
                fabric.print(f"{basename} Dice: {batch_dice:.4f} - ASSD: {batch_assd:.4f} - HD95: {batch_hd95:.4f}")

            torch.cuda.empty_cache()

    fabric.print(
        f'Validation [{iters}]: Mean Dice: [{dice_scores.avg:.4f}]'
        f' -- Mean ASSD: [{assd_scores.avg:.4f}]'
        f' -- Mean HD95: [{hd95_scores.avg:.4f}]'
    )
    case_results.append({
        "basename": "Average",
        "Dice": dice_scores.avg,
        "ASSD": assd_scores.avg,
        "HD95": hd95_scores.avg,
    })

    df = pd.DataFrame(case_results)
    if fabric.global_rank == 0:
        csv_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-test-results.csv")
        df.to_csv(csv_path, index=False)

    model.train()
    return dice_scores.avg, assd_scores.avg, hd95_scores.avg


def validate_med(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    model.eval()
    dice_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, _ = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)
            _, pred_masks, _, _ = model(images, prompts)

            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                pred_mask = (pred_mask > 0.5).cpu().numpy()
                gt_mask = (gt_mask > 0.5).cpu().numpy()
                batch_dice = metric.binary.dc(pred_mask, gt_mask)
                dice_scores.update(batch_dice, num_images)

            fabric.print(f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]: Mean Dice: [{dice_scores.avg:.4f}]')
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean Dice: [{dice_scores.avg:.4f}]')

    csv_dict = {
        "Name": name,
        "Prompt": cfg.prompt,
        "Mean Dice": f"{dice_scores.avg:.4f}",
        "iters": iters,
    }
    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_dict, csv_head=cfg.csv_keys)

    model.train()
    return dice_scores.avg


def validate(fabric: L.Fabric, cfg: Box, model: Model, val_dataloader: DataLoader, name: str, iters: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks, _ = data
            num_images = images.size(0)

            prompts = get_prompts(cfg, bboxes, gt_masks)
            _, pred_masks, _, _ = model(images, prompts)

            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(pred_mask, gt_mask.int(), mode='binary', threshold=0.5)
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)

            fabric.print(
                f'Val: [{iters}] - [{iter}/{len(val_dataloader)}]:'
                f' Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

    fabric.print(f'Validation [{iters}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')
    csv_dict = {
        "Name": name,
        "Prompt": cfg.prompt,
        "Mean IoU": f"{ious.avg:.4f}",
        "Mean F1": f"{f1_scores.avg:.4f}",
        "iters": iters,
    }
    if fabric.global_rank == 0:
        write_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_dict, csv_head=cfg.csv_keys)
    model.train()
    return ious.avg, f1_scores.avg
