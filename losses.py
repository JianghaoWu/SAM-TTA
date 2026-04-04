import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        return focal_loss.mean()


class DiceLoss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


def kl_bernoulli_loss(
    inputs,
    targets,
    temperature=1.0,
    size_average=True,
    multiply_T2=True,
    eps=1e-7,
    detach_target=False,
):
    """KL divergence loss between two Bernoulli distributions with temperature scaling."""
    logits = inputs / max(float(temperature), eps)
    q = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    p = targets.detach() if detach_target else targets
    p = p.clamp(eps, 1.0 - eps)

    q = q.view(-1)
    p = p.view(-1)

    kl = p * (torch.log(p) - torch.log(q)) + (1.0 - p) * (torch.log(1.0 - p) - torch.log(1.0 - q))

    if multiply_T2 and temperature is not None:
        kl = kl * (float(temperature) ** 2)

    return kl.mean() if size_average else kl.sum()


class ContraLoss(nn.Module):

    def __init__(self, temperature=0.3, weight=None, size_average=True):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, embedd_x: torch.Tensor, embedd_y: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor):
        x_embedding = self.norm_embed(embedd_x)
        y_embedding = self.norm_embed(embedd_y)

        x_masks = F.interpolate(mask_x, size=x_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()
        y_masks = F.interpolate(mask_y, size=y_embedding.shape[-2:], mode="bilinear", align_corners=False).detach()

        x_masks = torch.clamp(F.sigmoid(x_masks), min=0, max=1) > 0.5
        y_masks = torch.clamp(F.sigmoid(y_masks), min=0, max=1) > 0.5

        sum_x = x_masks.sum(dim=[-1, -2]).clone()
        sum_y = y_masks.sum(dim=[-1, -2]).clone()
        sum_x[sum_x[:, 0] == 0.] = 1.
        sum_y[sum_y[:, 0] == 0.] = 1.

        multi_embedd_x = (x_embedding * x_masks).sum(dim=[-1, -2]) / sum_x
        multi_embedd_y = (y_embedding * y_masks).sum(dim=[-1, -2]) / sum_y

        flatten_x = multi_embedd_x.view(multi_embedd_x.size(0), -1)
        flatten_y = multi_embedd_y.view(multi_embedd_y.size(0), -1)
        similarity_matrix = F.cosine_similarity(flatten_x.unsqueeze(1), flatten_y.unsqueeze(0), dim=2)

        label_pos = torch.eye(x_masks.size(0)).bool().to(embedd_x.device)
        similarity_matrix = similarity_matrix / self.temperature
        loss = -torch.log(
            similarity_matrix.masked_select(label_pos).exp().sum() /
            similarity_matrix.exp().sum()
        )
        return loss

    def norm_embed(self, embedding: torch.Tensor):
        return F.normalize(embedding, dim=0, p=2)

    def add_background(self, masks):
        mask_union = torch.max(masks, dim=0).values
        mask_complement = ~mask_union
        return torch.cat((masks, mask_complement.unsqueeze(0)), dim=0)


class FeatureAlignLoss(nn.Module):
    def __init__(self, mode='mse'):
        super().__init__()
        assert mode in ['cosine', 'mse']
        self.mode = mode

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor, mask: torch.Tensor):
        C, H, W = feat_s.shape
        mask = F.interpolate(mask.float(), size=(H, W), mode="bilinear", align_corners=False)
        mask = mask > 0.5
        return F.mse_loss(feat_s, feat_t, reduction='mean')


@torch.no_grad()
def _prep_mask(mask, H, W, device):
    """Normalize mask to shape [H*W] float vector; returns None if mask is None."""
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask[0]
    if mask.dim() == 2:
        mask = mask.reshape(-1)
    elif mask.dim() != 1:
        raise ValueError("mask must be [H,W], [1,H,W] or [H*W]")
    return mask.to(device=device, dtype=torch.float32)


def kl_spatial_per_channel(
    teacher: torch.Tensor,
    student: torch.Tensor,
    temp: float = 2.0,
    reduction: str = "mean",
    mask: torch.Tensor = None,
    symmetric_js: bool = False,
    eps: float = 1e-12,
):
    """
    Per-channel spatial KL divergence between teacher and student feature maps.

    Treats each channel's spatial activations as a probability distribution via
    softmax, then computes KL(P_teacher || P_student).

    Args:
        teacher, student: [C, H, W] feature maps
        temp: temperature for softmax (higher = softer distribution)
        reduction: 'mean' | 'sum' | 'none'
        mask: optional spatial mask [H,W] or [1,H,W]
        symmetric_js: if True, use symmetric Jensen-Shannon divergence
        eps: numerical stability constant
    """
    assert teacher.shape == student.shape and teacher.dim() == 3
    C, H, W = teacher.shape
    device = teacher.device

    t = teacher.reshape(C, -1) / max(temp, eps)
    s = student.reshape(C, -1) / max(temp, eps)

    t_logp = F.log_softmax(t, dim=1)
    s_logp = F.log_softmax(s, dim=1)
    t_prob = t_logp.exp()
    s_prob = s_logp.exp()

    m = _prep_mask(mask, H, W, device)

    def _kl(p_logp, q_logp, p_prob, m):
        if m is not None:
            p_prob_masked = p_prob * m[None, :]
            denom = p_prob_masked.sum(dim=1, keepdim=True).clamp_min(eps)
            p_prob_norm = p_prob_masked / denom
            kl_pos = p_prob_norm * (p_logp - q_logp) * m[None, :]
        else:
            kl_pos = p_prob * (p_logp - q_logp)
        return kl_pos.sum(dim=1)

    if symmetric_js:
        kl_c = 0.5 * (_kl(t_logp, s_logp, t_prob, m) + _kl(s_logp, t_logp, s_prob, m))
    else:
        kl_c = _kl(t_logp, s_logp, t_prob, m)

    if reduction == "mean":
        return kl_c.mean()
    elif reduction == "sum":
        return kl_c.sum()
    elif reduction == "none":
        return kl_c
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")


class SoftAlignLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor, iou_score: torch.Tensor):
        """
        Args:
            feat_s, feat_t: [C, H, W] student and teacher features
            iou_score: scalar IoU used as alignment weight
        """
        feat_s_map = feat_s.norm(p=2, dim=0)
        feat_t_map = feat_t.norm(p=2, dim=0)
        diff_map = torch.abs(feat_s_map - feat_t_map)
        weight = torch.exp(iou_score.to(feat_s.device))
        loss_map = diff_map * weight

        if self.reduction == 'mean':
            return loss_map.mean()
        elif self.reduction == 'sum':
            return loss_map.sum()
        return loss_map


class SoftAlignLossWithTemperature(nn.Module):
    def __init__(self, temp_mode='exp', k=2.0, reduction='mean', eps=1e-6):
        """
        Args:
            temp_mode: 'linear' | 'exp' | 'softplus'
            k: controls temperature scale for 'exp' mode
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.temp_mode = temp_mode
        self.k = k
        self.reduction = reduction
        self.eps = eps

    def compute_temperature(self, iou: torch.Tensor):
        iou = torch.clamp(iou, min=self.eps, max=1.0)
        if self.temp_mode == 'linear':
            return torch.clamp(1.0 - iou, min=self.eps)
        elif self.temp_mode == 'exp':
            return torch.exp(-self.k * iou)
        elif self.temp_mode == 'softplus':
            return F.softplus(1.0 - iou)
        else:
            raise NotImplementedError(f"Unknown temp_mode: {self.temp_mode}")

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor, iou_score: torch.Tensor):
        feat_s_map = feat_s.norm(p=2, dim=0)
        feat_t_map = feat_t.norm(p=2, dim=0)
        diff_map = torch.abs(feat_s_map - feat_t_map)
        tau = self.compute_temperature(iou_score.to(feat_s.device))
        loss_map = diff_map / tau

        if self.reduction == 'mean':
            return loss_map.mean()
        elif self.reduction == 'sum':
            return loss_map.sum()
        return loss_map
