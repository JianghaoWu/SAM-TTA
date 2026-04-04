import os
import csv
import copy

import numpy as np
import torch
import torch.nn.functional as F


def freeze(model: torch.nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def momentum_update(student_model, teacher_model, momentum=0.99):
    """Exponential moving average update: teacher = momentum * teacher + (1-momentum) * student."""
    for (src_name, src_param), (tgt_name, tgt_param) in zip(
        student_model.named_parameters(), teacher_model.named_parameters()
    ):
        if src_param.requires_grad:
            tgt_param.data.mul_(momentum).add_(src_param.data, alpha=1 - momentum)


def decode_mask(mask):
    """
    Convert a single-channel instance mask [1, H, W] with integer labels
    into a multi-channel binary mask [N, H, W].
    """
    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    new_mask = torch.zeros((len(unique_labels), *mask.shape[1:]), dtype=torch.int64)
    for i, label in enumerate(unique_labels):
        new_mask[i] = (mask == label).squeeze(0)
    return new_mask


def encode_mask(mask):
    """
    Convert a multi-channel binary mask [N, H, W] into a single-channel
    instance mask [1, H, W] with integer labels 1..N.
    """
    new_mask = torch.zeros((1, *mask.shape[1:]), dtype=torch.int64)
    for i in range(mask.shape[0]):
        new_mask[0][mask[i] == 1] = i + 1
    return new_mask


def copy_model(model: torch.nn.Module):
    """Deep copy a model and freeze all its parameters."""
    new_model = copy.deepcopy(model)
    freeze(new_model)
    return new_model


def create_csv(filename, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    if os.path.exists(filename):
        return
    with open(filename, 'w') as csvfile:
        csv.DictWriter(csvfile, fieldnames=csv_head).writeheader()


def write_csv(filename, csv_dict, csv_head=["corrupt", "Mean IoU", "Mean F1", "epoch"]):
    with open(filename, 'a+') as csvfile:
        csv.DictWriter(csvfile, fieldnames=csv_head, extrasaction='ignore').writerow(csv_dict)


def check_grad(model: torch.nn.Module):
    """Print which parameters require gradients."""
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


def reduce_instances(bboxes, gt_masks, max_nums=50):
    """Randomly subsample instances per image to at most max_nums."""
    bboxes_ = []
    gt_masks_ = []
    for bbox, gt_mask in zip(bboxes, gt_masks):
        idx = np.arange(bbox.shape[0])
        np.random.shuffle(idx)
        bboxes_.append(bbox[idx[:max_nums]])
        gt_masks_.append(gt_mask[idx[:max_nums]])
    return bboxes_, gt_masks_


def simple_uncertainty_entropy(pred_list):
    """
    Compute normalized binary entropy for a list of prediction tensors.

    Args:
        pred_list: list of torch.Tensor, each [1, H, W] or [1, N, H, W], values in [0, 1]

    Returns:
        list of torch.Tensor [1, H, W] with entropy normalized to [0, 1]
    """
    out_list = []
    eps = 1e-8
    log2 = torch.log(torch.tensor(2.0))

    for preds in pred_list:
        preds = preds.detach()
        inputs = F.sigmoid(preds)
        if inputs.dim() == 3:
            mean_p = inputs
        elif inputs.dim() == 4:
            mean_p = inputs.mean(dim=1)
        else:
            raise ValueError("Input must be [1,H,W] or [1,N,H,W]")

        p = mean_p.clamp(eps, 1 - eps)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        out_list.append(entropy / log2)

    return out_list
