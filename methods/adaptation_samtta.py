import argparse
import importlib
import os
import random
import sys
import time

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg
from datasets import call_load_dataset
from losses import DiceLoss, kl_spatial_per_channel
from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import AverageMeter, calculate_metrics, get_prompts
from utils.nonlinear_net import DynamicBezierTransform2D
from utils.tools import copy_model, create_csv, momentum_update, reduce_instances


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    anchor_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    all_dataloader: DataLoader,
    num_iters: int,
):
    """SAM-TTA training loop: test-time adaptation with learnable Bezier transforms."""
    data_time = AverageMeter()
    batch_time = AverageMeter()
    dice_loss = DiceLoss()
    end = time.time()
    num_epochs = 1  # TTA runs for a single pass over the test set

    dice_scores = AverageMeter()
    assd_scores = AverageMeter()
    hd95_scores = AverageMeter()
    iou_losses = AverageMeter()
    dice_losses = AverageMeter()
    ent_losses = AverageMeter()
    total_losses = AverageMeter()
    case_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_auto = DynamicBezierTransform2D(degree=3, num_curves=3).to(device)
    optimizer_auto = optim.Adam(transform_auto.parameters(), lr=0.1)

    w_max = 0.0
    for epoch in range(num_epochs):
        for iter, data in enumerate(all_dataloader):
            data_time.update(time.time() - end)
            images_test, bboxes_test, bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data

            transformed_images, _ = transform_auto(images_test[0])
            transformed_images = transformed_images.unsqueeze(0)

            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            prompts = get_prompts(cfg, bboxes_test, gt_masks_test)
            with torch.no_grad():
                anchor_image_embeds, anchor_masks, anchor_iou_predictions, anchor_res_masks = anchor_model(transformed_images, prompts)
            pred_image_embeds, pred_masks, pred_iou_predictions, pred_res_masks = model(transformed_images, prompts)

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_ent = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)

            for i, (pred_mask, anchor_mask, iou_prediction, anchor_res, pred_res) in enumerate(
                zip(pred_masks, anchor_masks, pred_iou_predictions, anchor_res_masks, pred_res_masks)
            ):
                iou_prediction = torch.clamp(iou_prediction, min=0, max=1)
                iou_score = iou_prediction.mean()

                loss_ent += kl_spatial_per_channel(
                    anchor_image_embeds[i].detach(), pred_image_embeds[i], temp=iou_score.detach()
                )
                anchor_mask = (anchor_mask > 0.).float()
                anchor_res = (anchor_res > 0.).float()

                eps = 1e-6
                w = -torch.log(torch.clamp(1.0 - iou_score.detach(), min=eps))
                if w > w_max:
                    w_max = w
                w_i = w / w_max
                loss_dice += dice_loss(pred_mask, anchor_mask) * w_i
                loss_dice += dice_loss(pred_res, anchor_res) * w_i

                loss_iou += (1.0 - iou_prediction.mean())

            loss_total = loss_iou + loss_dice / 2.0 + loss_ent

            fabric.backward(loss_total)
            optimizer.step()
            optimizer_auto.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_auto.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()
            momentum_update(model, anchor_model, momentum=0.95)

            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            ent_losses.update(loss_ent.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(
                f'Epoch: [{epoch}][{iter+1}/{len(all_dataloader)}]'
                f' | Dataset: [{cfg.dataset} - {cfg.prompt}]'
                f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                f' | Entropy Loss [{ent_losses.val:.4f} ({ent_losses.avg:.4f})]'
                f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]'
            )
            fabric.log_dict({
                "Dice Loss": dice_losses.avg,
                "IoU Loss": iou_losses.avg,
                "Total Loss": total_losses.avg,
            })
            torch.cuda.empty_cache()

            # Evaluate on current test sample
            num_images = transformed_images.size(0)
            with torch.no_grad():
                model.eval()
                transformed_images, _ = transform_auto(images_test[0], visual=False)
                transformed_images = transformed_images.unsqueeze(0)
                _, pred_masks, ious, _ = model(transformed_images, prompts)

                for pred_mask, gt_mask, basename, iou in zip(pred_masks, gt_masks_test, basenames, ious):
                    pred_mask = (pred_mask > 0.5).float()
                    gt_mask = (gt_mask > 0.5).float()
            model.train()

            batch_dice, batch_assd, batch_hd95, batch_iou = calculate_metrics(pred_mask, gt_mask)
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

    fabric.print(
        f'Test Ending: Mean Dice: [{dice_scores.avg:.4f}]'
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

    state = {"model": model, "optimizer": optimizer}
    fabric.save(os.path.join(cfg.out_dir, "save-ckpt", f"{cfg.dataset}-{cfg.prompt}-last-ckpt.pth"), state)


def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        return 0.999 ** step

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(
        accelerator="auto",
        devices=num_devices,
        strategy="auto",
        loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")]
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        cfg_dict_path = os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(cfg_dict_path, "w") as f:
            yaml.dump(cfg.to_dict(), f)
        os.makedirs(os.path.join(cfg.out_dir, "save-ckpt"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup_pm()
        LoRA_Sam(model.model, 4)
        anchor_model = copy_model(model)

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model.model)

    fabric.print(
        f"All: {len(all_data) * cfg.batch_size} | Train: {len(train_data) * cfg.batch_size}"
        f" | Val: {len(val_data) * cfg.val_batchsize} | Test: {len(test_data)}"
    )
    num_iters = len(train_data) * cfg.batch_size

    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])

    all_data = fabric._setup_dataloader(all_data)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, anchor_model, optimizer, scheduler, all_data, num_iters)
    del model, anchor_model, train_data, val_data


def load_config(cfg_path: str, args: argparse.Namespace) -> Box:
    config_module = importlib.import_module(cfg_path.replace('.py', '').replace('/', '.'))
    cfg = Box(config_module.config)
    if hasattr(config_module, 'base_config'):
        cfg.merge_update(config_module.base_config)
    if args.dataset:
        cfg.dataset = args.dataset
    if args.prompt:
        cfg.prompt = args.prompt
    if args.gpu_ids:
        cfg.gpu_ids = args.gpu_ids
    cfg.out_dir = "output/samtta/" + args.dataset
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-TTA: Test-Time Adaptation for SAM")
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file (e.g., configs/config_brats.py)')
    parser.add_argument('--dataset', type=str, help='Dataset name (overrides config)')
    parser.add_argument('--prompt', type=str, help='Prompt type: box or point (overrides config)')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs (overrides config)')
    args = parser.parse_args()

    set_seed(1337)
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
