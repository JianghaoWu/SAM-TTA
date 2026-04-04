import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_checkpoint(self, model_type):
        checkpoints = {
            "vit_b": "sam_vit_b_01ec64.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_h": "sam_vit_h_4b8939.pth",
            "medsam": "medsam_vit_b.pth",
        }
        if model_type not in checkpoints:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(checkpoints)}")
        return os.path.join(self.cfg.model.checkpoint, checkpoints[model_type])

    def setup(self):
        """Load checkpoint and apply freeze settings from config."""
        checkpoint = self.get_checkpoint(self.cfg.model.type)
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)
        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def setup_pm(self):
        """Load checkpoint with prompt encoder kept trainable."""
        checkpoint = self.get_checkpoint(self.cfg.model.type)
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)
        self.model.train()
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = True
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    def forward(self, images, prompts):
        image_embeddings = self.encode(images)
        pred_masks, ious, res_masks = self.decode(prompts, image_embeddings)
        return image_embeddings, pred_masks, ious, res_masks

    def encode(self, images):
        _, _, H, W = images.shape
        self.image_shape = (H, W)
        return self.model.image_encoder(images)

    def decode(self, prompts, image_embeddings):
        pred_masks, ious, res_masks = [], [], []
        for prompt, embedding in zip(prompts, image_embeddings):
            if isinstance(prompt, torch.Tensor):
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None, boxes=prompt.to(device=embedding.device), masks=None
                )
            elif isinstance(prompt, tuple):
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt, boxes=None, masks=None
                )
            else:
                raise ValueError(f"Unsupported prompt type: {type(prompt)}")

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = F.interpolate(low_res_masks, self.image_shape, mode="bilinear", align_corners=False)
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks

    def decode_from_embeds(self, image_embeddings, prompts):
        """Decode masks from pre-computed image embeddings (skips image encoder)."""
        pred_masks, ious, res_masks = [], [], []
        for prompt, embedding in zip(prompts, image_embeddings):
            if isinstance(prompt, torch.Tensor):
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None, boxes=prompt.to(device=embedding.device), masks=None
                )
            elif isinstance(prompt, tuple):
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt, boxes=None, masks=None
                )
            else:
                raise ValueError(f"Unsupported prompt type: {type(prompt)}")

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = F.interpolate(low_res_masks, self.image_shape, mode="bilinear", align_corners=False)
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks

    def get_predictor(self):
        return SamPredictor(self.model)

    def get_generator(self, output_mode):
        return SamAutomaticMaskGenerator(self.model, output_mode=output_mode)
