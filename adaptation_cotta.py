import argparse
import importlib.util
import os
import yaml
import time
import torch
import lightning as L
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from box import Box
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
from copy import deepcopy
import PIL

from configs.config import cfg
from losses import DiceLoss, FocalLoss, ContraLoss
from datasets import call_load_dataset

from model import Model
from sam_lora import LoRA_Sam
from utils.eval_utils import AverageMeter, calc_iou, validate, get_prompts, validate_med, test_med, calculate_metrics
from utils.tools import copy_model, create_csv, check_grad, momentum_update, reduce_instances
from PIL import Image
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from PIL import Image, ImageDraw
import numpy as np

from PIL import Image, ImageDraw
import numpy as np

from PIL import Image, ImageDraw
import numpy as np
import torch

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

import torchvision.transforms as transforms
import utils.my_transform as my_transforms
def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (1024, 1024, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0), 
        # my_transforms.ColorJitterPro(
        #     brightness=[0.8, 1.2] if soft else [0.6, 1.4],
        #     contrast=[0.85, 1.15] if soft else [0.7, 1.3],
        #     saturation=[0.75, 1.25] if soft else [0.5, 1.5],
        #     hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
        #     gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        # ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  
        # transforms.RandomAffine(
        #     degrees=[-8, 8] if soft else [-15, 15],
        #     translate=(1/16, 1/16),
        #     scale=(0.95, 1.05) if soft else (0.9, 1.1),
        #     shear=None,
        #     resample=PIL.Image.BILINEAR,
        #     fillcolor=None
        # ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    ema_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    all_dataloader: DataLoader,
    num_iters: int,
):
    """The SAM training loop."""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    focal_losses = AverageMeter()
    dice_losses = AverageMeter()
    iou_losses = AverageMeter()
    anchor_losses = AverageMeter()
    contra_losses = AverageMeter()
    total_losses = AverageMeter()
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    contra_loss = ContraLoss()
    end = time.time()
    max_dice = 0.
    num_epochs = 1  ## number of epochs, TTA was set to 1
    #### uese for testing purposes
    dice_scores = AverageMeter()
    assd_scores = AverageMeter() 
    hd95_scores = AverageMeter() 
    case_results = []
    anchor_model = copy_model(model)
    transform_cotta = get_tta_transforms()  
    model_state = deepcopy(model.state_dict())
    

    for epoch in range(0, num_epochs):
        for iter, data in enumerate(all_dataloader):
            data_time.update(time.time() - end)
            # images_test, bboxes_test, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data
            images_test, bboxes_test,bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data

            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                print(num_insts)
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            prompts = get_prompts(cfg, bboxes, gt_masks)

            with torch.no_grad():
                standard_ema_image_embeds, standard_ema_masks, standard_ema_iou_predictions, standard_ema_res_masks = ema_model(images_test, prompts)
                anchor_image_embeds, anchor_masks, anchor_iou_predictions, anchor_res_masks = anchor_model(images_test, prompts)
            pred_image_embeds, pred_masks, iou_predictions, pred_res_masks = model(images_test, prompts)   # student

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            # N = 32 
            outputs_emas = []
            for i in (images_test, images_weak, images_strong):
                outputs_  = ema_model(i, prompts)
                outputs_emas.append(outputs_[0].detach())

            loss_iou = torch.tensor(0., device=fabric.device)

            for i, (standard_ema_mask, anchor_mask, pred_mask) in enumerate(zip(standard_ema_masks, anchor_masks, pred_masks)):
                anchor_mask_sig = F.sigmoid(anchor_mask)
                anchor_prob = anchor_mask_sig.max()  # 直接取最大值

                print(anchor_prob.mean(), '180')
                
                if anchor_prob.mean()<0.92:
                    outputs_ema = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_ema = standard_ema_mask
                
                # loss = (softmax_entropy(pred_mask, outputs_ema)).mean(0) 


                outputs_ema = (outputs_ema > 0.).float()

                loss_iou += dice_loss(pred_mask, outputs_ema)


            loss_total = loss_iou
            fabric.backward(loss_total)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()

            momentum_update(model, ema_model, momentum=cfg.ema_rate)

            if True:
                # 检查模型状态字典中的键
                # 打印出只有需要梯度的参数的键
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(name)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # 去除 _forward_module 前缀
                        cleaned_key = name.replace("_forward_module.", "")
                        
                        # 确保在 model_state 中找到对应的键
                        if cleaned_key in model_state:
                            mask = (torch.rand(param.shape) < 0.02).float().cuda()
                            with torch.no_grad():
                                param.data = model_state[cleaned_key] * mask + param * (1. - mask)
                        else:
                            print(f"Parameter key {cleaned_key} not found in model_state.")

            anchor_losses.update(loss_iou.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(all_dataloader)}]'
                         f' | Dataset: [{cfg.dataset} - {cfg.prompt}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Anchor Loss [{anchor_losses.val:.4f} ({anchor_losses.avg:.4f})]')

            loss_logger = {"Anchor Loss": anchor_losses.avg}
            fabric.log_dict(loss_logger)
            torch.cuda.empty_cache()
            ############################### Test the final result #################################
            model.eval()
            num_images = images_test.size(0)
            prompts = get_prompts(cfg, bboxes_test, gt_masks_test)

            _, pred_masks, _, _ = model(images_test, prompts)
            for pred_mask, gt_mask, basename in zip(pred_masks, gt_masks_test, basenames):
                pred_mask = (pred_mask > 0.5).float()
                gt_mask = (gt_mask > 0.5).float()

                batch_dice, batch_assd, batch_hd95 = calculate_metrics(pred_mask, gt_mask)

                dice_scores.update(batch_dice, num_images)
                assd_scores.update(batch_assd, num_images)
                hd95_scores.update(batch_hd95, num_images)

                pred_mask_np = pred_mask.squeeze().cpu().numpy()  
                pred_mask_np = (pred_mask_np * 255).astype(np.uint8)  

                save_path = os.path.join(cfg.out_dir, f"{cfg.dataset}-{cfg.prompt}-pred_masks", f"{basename}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(pred_mask_np.shape,'pred_mask_np')

                Image.fromarray(pred_mask_np).save(save_path)

                case_results.append({
                    "basename": basename, 
                    "Dice": batch_dice, 
                    "ASSD": batch_assd, 
                    "HD95": batch_hd95
                })
                

                fabric.print(f"{basename} Dice: {batch_dice:.4f} - Batch ASSD: {batch_assd:.4f} - Batch HD95: {batch_hd95:.4f}")
            model.train()
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
    cfg.out_dir = "output/cotta_tta/"+args.dataset

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
