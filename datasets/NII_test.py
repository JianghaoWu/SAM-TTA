import os
import cv2
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from skimage.draw import polygon2mask

from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_, decode_mask


def expand_bbox(x, y, w, h, width, height, expand_ratio=0.25):
    """Expand a bounding box by expand_ratio on each side, clamped to image bounds."""
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    new_x = max(0, x - (new_w - w) / 2)
    new_y = max(0, y - (new_h - h) / 2)
    new_x2 = min(width, x + w + (new_w - w) / 2)
    new_y2 = min(height, y + h + (new_h - h) / 2)
    return new_x, new_y, new_x2, new_y2


class NIIDataset(Dataset):
    """
    Dataset for 2D slices stored as NIfTI (.nii / .nii.gz) files.

    Supports test-time self-training via the if_self_training flag, which
    returns both a test image and weakly/strongly augmented views.
    """

    def __init__(self, cfg, root_dir, list_file, transform=None, if_self_training=False):
        self.cfg = cfg
        df = pd.read_csv(list_file, encoding='gbk')
        self.name_list = df.iloc[:, 0].tolist()
        self.label_list = df.iloc[:, 1].tolist()
        self.root_dir = root_dir
        self.transform = transform
        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        basename = os.path.splitext(os.path.basename(name))[0]
        image_path = os.path.join(self.root_dir, name)

        img_nii = nib.load(image_path)
        image = img_nii.get_fdata()

        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
        elif image.ndim == 3:
            image = image[0, :, :]
            image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        height, width, _ = image.shape

        if self.cfg.get_prompt:
            return idx, {"file_path": image_path, "height": height, "width": width}, image

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)
        gt_nii = nib.load(gt_path)
        gt_mask = (gt_nii.get_fdata() * 255).astype(np.uint8)

        masks, bboxes, bboxes_coarse, categories = [], [], [], []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()

        for mask in gt_masks:
            x, y, w, h = cv2.boundingRect(mask)
            bbox = list(expand_bbox(x, y, w, h, width, height, expand_ratio=0.0))
            masks.append(mask)
            bboxes.append(bbox)
            bboxes_coarse.append(bbox)
            categories.append("0")

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)
                image_orig, masks_orig = image, [m for m in masks]
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
                _, _, bboxes_coarse = self.transform(image_orig, masks_orig, np.array(bboxes_coarse))

            return (
                image,
                torch.tensor(np.stack(bboxes, axis=0)),
                torch.tensor(np.stack(bboxes_coarse, axis=0)),
                torch.tensor(np.stack(masks, axis=0)).float(),
                basename,
                image_weak,
                image_strong,
                torch.tensor(np.stack(bboxes_weak, axis=0)),
                torch.tensor(np.stack(masks_weak, axis=0)).float(),
            )

        elif self.cfg.visual:
            origin_image = image
            origin_bboxes = bboxes.copy()
            origin_masks = masks.copy()
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)
            return (
                os.path.splitext(os.path.basename(name))[0],
                padding,
                origin_image,
                np.stack(origin_bboxes, axis=0),
                np.stack(origin_masks, axis=0),
                image,
                torch.tensor(np.stack(bboxes, axis=0)),
                torch.tensor(np.stack(masks, axis=0)).float(),
            )

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            return (
                image,
                torch.tensor(np.stack(bboxes, axis=0)),
                torch.tensor(np.stack(masks, axis=0)).float(),
                basename,
            )


class NIIDatasetwithCoarse(NIIDataset):
    """NIIDataset variant that uses polygon-approximated coarse bounding boxes."""

    def __getitem__(self, idx):
        name = self.name_list[idx]
        basename = os.path.splitext(os.path.basename(name))[0]

        image_path = os.path.join(self.root_dir, name)
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        elif image.ndim == 3 and image.shape[0] != 3:
            image = np.moveaxis(image, 0, -1)
            if image.shape[-1] != 3:
                image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        label_name = self.label_list[idx]
        gt_path = os.path.join(self.root_dir, label_name)
        gt_nii = nib.load(gt_path)
        gt_mask = gt_nii.get_fdata()
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[0, :, :]
        gt_mask = (gt_mask * 255).astype(np.uint8)

        masks, bboxes, approxes, categories = [], [], [], []
        gt_masks = decode_mask(torch.tensor(gt_mask[None, :, :])).numpy().astype(np.uint8)
        assert gt_masks.sum() == (gt_mask > 0).sum()

        for mask in gt_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_vertices = max(0.05 * cv2.arcLength(contours[0], True), 3)
            approx = cv2.approxPolyDP(contours[0], num_vertices, True).squeeze(1)
            coords = np.array(approx)
            x_max, x_min = coords[:, 0].max(), coords[:, 0].min()
            y_max, y_min = coords[:, 1].max(), coords[:, 1].min()

            if x_min == x_max or y_min == y_max:
                x, y, w, h = cv2.boundingRect(mask)
                bboxes.append([x, y, x + w, y + h])
            else:
                bboxes.append([x_min, y_min, x_max, y_max])

            masks.append(mask)
            categories.append("0")
            approxes.append(approx)

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)
            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)
            return (
                image_weak,
                image_strong,
                torch.tensor(np.stack(bboxes_weak, axis=0)),
                torch.tensor(np.stack(masks_weak, axis=0)).float(),
            )

        elif self.cfg.visual:
            origin_image = image
            origin_approxes = approxes
            origin_masks = masks.copy()
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), self.cfg.visual)
            return (
                os.path.splitext(os.path.basename(name))[0],
                padding,
                origin_image,
                origin_approxes,
                np.stack(origin_masks, axis=0),
                image,
                torch.tensor(np.stack(bboxes, axis=0)),
                torch.tensor(np.stack(masks, axis=0)).float(),
            )

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            return (
                image,
                torch.tensor(np.stack(bboxes, axis=0)),
                torch.tensor(np.stack(masks, axis=0)).float(),
                basename,
            )


def _make_dataloader(dataset, batch_size, shuffle, num_workers, collate):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate)


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    dataset_name = cfg.dataset
    root = cfg.datasets[dataset_name].root_dir

    all_ds = NIIDataset(cfg, root, cfg.datasets[dataset_name].all_list, transform, if_self_training=cfg.augment)
    train_ds = NIIDataset(cfg, root, cfg.datasets[dataset_name].train_list, transform, if_self_training=cfg.augment)
    val_ds = NIIDataset(cfg, root, cfg.datasets[dataset_name].val_list, transform)
    test_ds = NIIDataset(cfg, root, cfg.datasets[dataset_name].test_list, transform, if_self_training=cfg.augment)

    return (
        _make_dataloader(all_ds, cfg.batch_size, True, cfg.num_workers, collate_fn),
        _make_dataloader(train_ds, cfg.batch_size, True, cfg.num_workers, collate_fn),
        _make_dataloader(val_ds, cfg.val_batchsize, True, cfg.num_workers, collate_fn),
        _make_dataloader(test_ds, 1, True, cfg.num_workers, collate_fn),
    )


def load_datasets_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    dataset_name = cfg.dataset
    root = cfg.datasets[dataset_name].root_dir

    train_ds = NIIDatasetwithCoarse(cfg, root, cfg.datasets[dataset_name].train_list, transform, if_self_training=cfg.augment)
    val_ds = NIIDatasetwithCoarse(cfg, root, cfg.datasets[dataset_name].val_list, transform)
    test_ds = NIIDatasetwithCoarse(cfg, root, cfg.datasets[dataset_name].test_list, transform)

    return (
        _make_dataloader(train_ds, cfg.batch_size, True, cfg.num_workers, collate_fn),
        _make_dataloader(val_ds, cfg.val_batchsize, True, cfg.num_workers, collate_fn),
        _make_dataloader(test_ds, 1, True, cfg.num_workers, collate_fn),
    )


def load_datasets_visual(cfg, img_size):
    transform = ResizeAndPad(img_size)
    dataset_name = cfg.dataset
    val_ds = NIIDataset(cfg, cfg.datasets[dataset_name].root_dir, cfg.datasets[dataset_name].test_list, transform)
    return _make_dataloader(val_ds, cfg.val_batchsize, True, cfg.num_workers, collate_fn_)


def load_datasets_visual_coarse(cfg, img_size):
    transform = ResizeAndPad(img_size)
    dataset_name = cfg.dataset
    val_ds = NIIDatasetwithCoarse(cfg, cfg.datasets[dataset_name].root_dir, cfg.datasets[dataset_name].test_list, transform)
    return _make_dataloader(val_ds, cfg.val_batchsize, True, cfg.num_workers, collate_fn_)


def load_datasets_prompt(cfg, img_size):
    transform = ResizeAndPad(img_size)
    dataset_name = cfg.dataset
    train_ds = NIIDataset(cfg, cfg.datasets[dataset_name].root_dir, cfg.datasets[dataset_name].train_list, transform, if_self_training=cfg.augment)
    return _make_dataloader(train_ds, cfg.batch_size, True, cfg.num_workers, collate_fn_)
