import argparse
import importlib
import os
import sys
import time

import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg
from datasets import call_load_dataset
from losses import DiceLoss
from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import AverageMeter, calculate_metrics, get_prompts
from utils.tools import copy_model, create_csv, reduce_instances


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
    """MedSAM source-only baseline: inference without test-time adaptation."""
    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()
    num_epochs = 1

    dice_scores = AverageMeter()
    assd_scores = AverageMeter()
    hd95_scores = AverageMeter()
    case_results = []

    for epoch in range(num_epochs):
        for iter, data in enumerate(all_dataloader):
            data_time.update(time.time() - end)
            images_test, bboxes_test, bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data

            model.eval()
            num_images = images_test.size(0)
            prompts = get_prompts(cfg, bboxes_test, gt_masks_test)

            _, pred_masks, _, _ = model(images_test, prompts)
            for pred_mask, gt_mask, basename in zip(pred_masks, gt_masks_test, basenames):
                pred_mask = (pred_mask > 0.5).float()
                gt_mask = (gt_mask > 0.5).float()

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

            batch_time.update(time.time() - end)
            end = time.time()

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
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

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
        model.setup()
        anchor_model = copy_model(model)
        LoRA_Sam(model.model, 4)

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model.model)

    fabric.print(
        f"All: {len(all_data) * cfg.batch_size} | Train: {len(train_data) * cfg.batch_size}"
        f" | Val: {len(val_data) * cfg.val_batchsize} | Test: {len(test_data)}"
    )
    num_iters = len(train_data) * cfg.batch_size

    if ckpt is not None:
        fabric.print(f"Loading checkpoint from {ckpt}")
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
    cfg.out_dir = "output/medsam_tta/" + args.dataset
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSAM source-only baseline")
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file (e.g., configs/config_medsam.py)')
    parser.add_argument('--dataset', type=str, help='Dataset name (overrides config)')
    parser.add_argument('--prompt', type=str, help='Prompt type: box or point (overrides config)')
    parser.add_argument('--gpu_ids', type=str, help='GPU IDs (overrides config)')
    args = parser.parse_args()
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
