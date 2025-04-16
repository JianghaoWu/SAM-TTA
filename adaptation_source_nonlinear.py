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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from math import comb

from utils.nonlinear import LearnableBezierTransform


    
def visualize_and_save_points_on_image(image, gt, po_points, na_points, filename):
    image = image * 255 
    image_np = image.permute(1, 2, 0).cpu().numpy()  
    gt_np = gt[0].cpu().numpy()  
    gt_np = gt_np[0]
    gt_np = np.uint8(gt_np * 255)  
    
    gt_pil = Image.fromarray(gt_np)
    gt_pil = gt_pil.resize((image_np.shape[1], image_np.shape[0])) 
    gt_np_resized = np.array(gt_pil)

    image_pil = Image.fromarray(np.uint8(image_np))
    draw = ImageDraw.Draw(image_pil)

    point_size = 5  

    po_points = po_points[0] 
    na_points = na_points[0]  

    for point in po_points:
        x, y = int(point[1]), int(point[0])
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill='black')

    for point in na_points:
        x, y = int(point[1]), int(point[0])
        draw.ellipse((x - point_size, y - point_size, x + point_size, y + point_size), fill='blue')

    gt_pil = Image.fromarray(gt_np_resized)
    combined_image = Image.new('RGB', (image_pil.width, image_pil.height + gt_pil.height))
    combined_image.paste(image_pil, (0, 0))
    combined_image.paste(gt_pil.convert('RGB'), (0, image_pil.height))

    combined_image.save(filename)



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

    end = time.time()
    max_dice = 0.
    num_epochs = 1  ## number of epochs, TTA was set to 1
    #### uese for testing purposes
    dice_scores = AverageMeter()
    assd_scores = AverageMeter() 
    hd95_scores = AverageMeter() 
    case_results = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # transform = LearnableBezierTransform(1024, 1024).to(device)
    transform = LearnableBezierTransform().to(device)
    optimizer = optim.Adam(transform.parameters(), lr=0.02)
    
    for epoch in range(0, num_epochs):
        for iter, data in enumerate(all_dataloader):
            loss = 0.
            data_time.update(time.time() - end)
            images_test, bboxes_test, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data
            ################################################
            # print(images_test.shape, len(images_test),images_test[0].shape, images_test.max(), images_test.min(), 'images_test')

            transformed_images = transform(images_test[0][0:1])  # 输入单通道 [1, 1024, 1024]
            transformed_images = transformed_images.unsqueeze(0)  # 扩展 batch 维度


            
            model.eval()
            num_images = transformed_images.size(0)
            prompts = get_prompts(cfg, bboxes_test, gt_masks_test)
            optimizer.zero_grad()
            _, pred_masks, ious, _ = model(transformed_images, prompts)

            for iou in ious:
                loss += 1.0 - iou
            loss.backward()
            optimizer.step()
            ################################
            with torch.no_grad():
                transformed_images = transform(images_test[0][0:1])  # 输入单通道 [1, 1024, 1024]
                transformed_images = transformed_images.unsqueeze(0)  # 扩展 batch 维度
                _, pred_masks, ious, _ = model(transformed_images, prompts)


                for pred_mask, gt_mask, basename, iou in zip(pred_masks, gt_masks_test, basenames, ious):
                    pred_mask = (pred_mask > 0.5).float()
                    gt_mask = (gt_mask > 0.5).float()

            print(f'{basename} - IoU: {iou}, Loss: {loss.item()}')

            batch_dice, batch_assd, batch_hd95 = calculate_metrics(pred_mask, gt_mask)

            dice_scores.update(batch_dice, num_images)
            assd_scores.update(batch_assd, num_images)
            hd95_scores.update(batch_hd95, num_images)

            pred_mask_np = pred_mask.squeeze().cpu().numpy()  
            pred_mask_np = (pred_mask_np * 255).astype(np.uint8)  

            save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basename}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            Image.fromarray(pred_mask_np).save(save_path)

            case_results.append({
                "basename": basename, 
                "Dice": batch_dice, 
                "ASSD": batch_assd, 
                "HD95": batch_hd95
            })
            

            fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")
    fabric.print(f'Test Ending...: Mean Dice: [{dice_scores.avg:.4f}] -- Mean ASSD: [{assd_scores.avg:.4f}] -- Mean HD95: [{hd95_scores.avg:.4f}]')
    case_results.append({
        "basename": "Average",  
        "Dice": dice_scores.avg,
        "ASSD": assd_scores.avg,
        "HD95": hd95_scores.avg
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
            return 1 / (cfg.opt.decay_factor**2)

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
        model.setup()
        anchor_model = copy_model(model)
        LoRA_Sam(model.model, 4)

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
    cfg.out_dir = "output/source-only_tta/"+args.dataset

    return cfg

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
