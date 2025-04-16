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


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=0)
    return entropy, avg_logits

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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    end = time.time()
    num_epochs = 1  ## number of epochs, TTA was set to 1
    #### uese for testing purposes
    dice_scores = AverageMeter()
    assd_scores = AverageMeter() 
    hd95_scores = AverageMeter() 
    case_results = []

    for epoch in range(0, num_epochs):
        for iter, data in enumerate(all_dataloader):
            data_time.update(time.time() - end)
            # images_test, bboxes_test, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data
            images_test, bboxes_test,bboxes_test_coarse, gt_masks_test, basenames, images_weak, images_strong, bboxes, gt_masks = data

            inputs = []
            inputs.append(images_test)
            inputs.append(images_strong)
            inputs = torch.stack(inputs)
            inputs = inputs.squeeze(dim=1)

            batch_size = inputs.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                print(num_insts)
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)

            prompts = get_prompts(cfg, bboxes, gt_masks)
            # prompts_expanded = prompts[0].repeat(2, 1)
            # prompts = (prompts_expanded,)
            print(prompts)
            if isinstance(prompts, list):
                # Assuming prompts[0] is a tensor of coordinates, we repeat it
                prompts_expanded = prompts[0].repeat(2, 1, 1)  # Repeat along the batch dimension (2 times)
                prompts = (prompts_expanded, prompts[1])  # Keeping the second tensor (which seems like a mask or label)
            else:
                prompts_expanded = prompts.repeat(2, 1)
                prompts = (prompts_expanded,)

            _, soft_masks, _, _ = model(inputs, prompts)    # teacher
            loss_ent = torch.tensor(0., device=fabric.device)
            
            for soft_mask in soft_masks:
                loss, logits = marginal_entropy(soft_mask)
                loss_ent += loss.mean() / 1.0

            fabric.backward(loss_ent)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()

            total_losses.update(loss_ent.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(all_dataloader)}]'
                         f' | Dataset: [{cfg.dataset} - {cfg.prompt}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            loss_logger = {"Total Loss": total_losses.avg}
            fabric.log_dict(loss_logger)
            torch.cuda.empty_cache()
            ############################### Test the final result #################################
            with torch.no_grad():
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
        anchor_model = copy_model(model)
        LoRA_Sam(model.model, 1)

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
    cfg.out_dir = "output/memo_tta/"+args.dataset

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
