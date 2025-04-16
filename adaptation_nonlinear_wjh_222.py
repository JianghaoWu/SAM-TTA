import argparse
import importlib.util
import os
import yaml
import time
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
import random
from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset

from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import AverageMeter, calc_iou, validate, get_prompts, validate_med, test_med, calculate_metrics
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
import pandas as pd
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from math import comb


from utils.nonlinear import LearnableBezierTransform


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
    """The SAM training loop."""
    data_time = AverageMeter()
    dice_loss = DiceLoss()
    end = time.time()
    max_dice = 0.
    num_epochs = 1  ## number of epochs, TTA was set to 1
    #### uese for testing purposes
    dice_scores = AverageMeter()
    assd_scores = AverageMeter() 
    hd95_scores = AverageMeter() 
    batch_time = AverageMeter()
    iou_losses = AverageMeter()
    dice_losses = AverageMeter()
    ent_losses = AverageMeter()
    total_losses = AverageMeter()
    case_results = []
    anchor_model_free = copy_model(anchor_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform_auto = LearnableBezierTransform().to(device)
    optimizer_auto = optim.Adam(transform_auto.parameters(), lr=0.01)
    def lr_lambda(step):
        # Decay rate for each step (0.999 for multiplicative decay)
        decay_rate = 0.99
        # Apply multiplicative decay at every step
        return decay_rate ** step
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_auto, lr_lambda)
    
    for epoch in range(0, num_epochs):
        t11 = time.time()
        ii = 0
        for iter, data in enumerate(all_dataloader):
            ii +=1
            data_time.update(time.time() - end)
            images_test, bboxes_test,bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data
            ################################################
            transformed_images = transform_auto(images_test[0][0:1], '66')  # 输入单通道 [1, 1024, 1024]
            transformed_images = transformed_images.unsqueeze(0)  # 扩展 batch 维度
            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                print(num_insts)
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)
            ################################################

            prompts = get_prompts(cfg, bboxes_test, gt_masks_test)
            prompts_coarse = get_prompts(cfg, bboxes_test_coarse, gt_masks_test)
            with torch.no_grad():
                anchor_image_embeds, anchor_masks, anchor_iou_predictions, anchor_res_masks = anchor_model(transformed_images, prompts)
                # _, anchor_masks_coarse, _, anchor_res_masks_coarse = anchor_model_free(transformed_images, prompts)
            pred_image_embeds, pred_masks, pred_iou_predictions, pred_res_masks = model(transformed_images, prompts)   
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_ent = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)

            for i, (pred_mask, anchor_mask, iou_prediction, anchor_res, pred_res) in enumerate(zip(pred_masks, anchor_masks, pred_iou_predictions, anchor_res_masks, pred_res_masks)):
                iou_score = iou_prediction.mean()
                anchor_mask = (anchor_mask > 0.).float()
                anchor_res = (anchor_res > 0.).float()
                loss_dice += dice_loss(pred_mask, anchor_mask)*(iou_score*iou_score*iou_score)
                loss_dice += dice_loss(pred_res, anchor_res)*(iou_score*iou_score*iou_score)
                loss_iou += (1.0 - iou_prediction.mean())

            loss_total = loss_iou + loss_dice *0.5

            fabric.backward(loss_total)

            optimizer.step()
            optimizer_auto.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_auto.zero_grad()
            torch.cuda.empty_cache()


            momentum_update(model, anchor_model, momentum=0.95)

                
            ################################
            # num_images = transformed_images.size(0)
            # with torch.no_grad():
            #     transformed_images = transform_auto(images_test[0][0:1],'66') 
            #     transformed_images = transformed_images.unsqueeze(0)  
            #     _, pred_masks, ious, _ = model(transformed_images, prompts)
        t22 = time.time()
        print(t22-t11, (t22-t11)/ii)
                



# def configure_opt(cfg: Box, model: Model):

#     def lr_lambda(step):
#         if step < cfg.opt.warmup_steps:
#             return step / cfg.opt.warmup_steps
#         elif step < cfg.opt.steps[0]:
#             return 1.0
#         elif step < cfg.opt.steps[1]:
#             return 1 / cfg.opt.decay_factor
#         else:
#             return 1 / (cfg.opt.decay_factor**2)
#     optimizer = torch.optim.Adam(
#             list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()), 
#             lr=cfg.opt.learning_rate, 
#             weight_decay=cfg.opt.weight_decay
#         )
#     # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#     return optimizer, scheduler

# def configure_opt(cfg: Box, model: Model):
#     def lr_lambda(step):
#         if step < cfg.opt.warmup_steps:
#             return step / cfg.opt.warmup_steps
#         elif step < cfg.opt.steps[0]:
#             return 1.0
#         elif step < cfg.opt.steps[1]:
#             return 1 / cfg.opt.decay_factor
#         else:
#             return 1 / (cfg.opt.decay_factor**2)

#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#     return optimizer, scheduler
def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        # Decay rate for each step (0.999 for multiplicative decay)
        decay_rate = 0.999
        
        # Apply multiplicative decay at every step
        return decay_rate ** step

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def multi_main(cfg):
    prompts = ["box", "point"]
    for prompt in prompts:
        cfg.prompt = prompt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box, ckpt: str = None) -> None:
    gpu_ids = cfg.gpu_ids.split(',')
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name=f"{cfg.dataset}-{cfg.prompt}")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        cfg_dict = cfg.to_dict()
        os.makedirs(os.path.join(cfg.out_dir, "configs"), exist_ok=True)
        cfg_dict_path = os.path.join(cfg.out_dir, "configs", f"{cfg.dataset}-{cfg.prompt}.yaml")
        with open(cfg_dict_path, "w") as file:
            yaml.dump(cfg_dict, file)

        os.makedirs(os.path.join(cfg.out_dir, "save-ckpt"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-training.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        
        # model.setup()
        model.setup_pm()
        LoRA_Sam(model.model, 4)
        anchor_model = copy_model(model)

    load_datasets = call_load_dataset(cfg)
    all_data, train_data, val_data, test_data = load_datasets(cfg, model.model.image_encoder.img_size)
    optimizer, scheduler = configure_opt(cfg, model.model)

    fabric.print(f"All Data: {len(all_data) * cfg.batch_size}; Train Data: {len(train_data) * cfg.batch_size}; Val Data: {len(val_data) * cfg.val_batchsize}; Val Data: {len(test_data) * 1}")
    num_iters = len(train_data) * cfg.batch_size
    if ckpt is not None:
        full_checkpoint = fabric.load(ckpt)
        model.load_state_dict(full_checkpoint["model"])
        # optimizer.load_state_dict(full_checkpoint["optimizer"])
    all_data = fabric._setup_dataloader(all_data)
    # train_data = fabric._setup_dataloader(train_data)
    # val_data = fabric._setup_dataloader(val_data)
    # test_data = fabric._setup_dataloader(test_data)
    model, optimizer = fabric.setup(model, optimizer)

    # validate_med(fabric, cfg, anchor_model, all_data, name=cfg.name, iters=0)

    train_sam(cfg, fabric, model, anchor_model, optimizer, scheduler, all_data, num_iters)

    del model, anchor_model, train_data, val_data

def load_config(cfg_path: str, args: argparse.Namespace):
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
    cfg.out_dir = "output/wjh_tta_nonl_001_iccv_new_metric_ABLATION/"+args.dataset

    return cfg
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多卡
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁止使用cudnn的自动优化

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg', type=str, required=True, help='Path to the configuration Python file (e.g., configs/config.py)')
    parser.add_argument('--dataset', type=str, help='Dataset name to override the config file')
    parser.add_argument('--prompt', type=str, help='Prompt type to override the config file')
    parser.add_argument('--gpu_ids', type=str, help='Gpu IDs to override the config file')
    args = parser.parse_args()
    cfg = load_config(args.cfg, args)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    main(cfg)
    torch.cuda.empty_cache()
    set_seed(1337)
