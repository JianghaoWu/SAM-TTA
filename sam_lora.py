"""
LoRA (Low-Rank Adaptation) for SAM's image encoder.

Applies low-rank decomposition to the QKV projection weights in each
attention block, enabling parameter-efficient fine-tuning.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """Wraps SAM's QKV linear layer with LoRA low-rank residuals for Q and V."""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """
    Applies LoRA to SAM's image encoder attention blocks.

    Freezes the original encoder weights and injects trainable low-rank
    matrices into all QKV projections.

    Args:
        sam_model: SAM model instance
        r: LoRA rank
        lora_layer: list of block indices to apply LoRA (default: all blocks)
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super().__init__()
        assert r > 0

        self.lora_layer = lora_layer or list(range(len(sam_model.image_encoder.blocks)))
        self.w_As = []
        self.w_Bs = []

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_q = nn.Linear(self.dim, r, bias=False)
            w_b_q = nn.Linear(r, self.dim, bias=False)
            w_a_v = nn.Linear(self.dim, r, bias=False)
            w_b_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.extend([w_a_q, w_a_v])
            self.w_Bs.extend([w_b_q, w_b_v])
            blk.attn.qkv = _LoRA_qkv(w_qkv_linear, w_a_q, w_b_q, w_a_v, w_b_v)

        self._reset_parameters()
        self.lora_vit = sam_model

    def _reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
