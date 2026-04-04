import numpy as np
import torch
import torch.nn as nn
from math import comb
from PIL import Image


class LearnableBezierTransform(nn.Module):
    """
    Learnable per-channel Bezier curve intensity transform.

    Three independent cubic Bezier curves are applied to the input image,
    producing a 3-channel output. Control points are learned end-to-end.

    Args:
        num_control_points: number of control points per curve (default 4 for cubic)
    """

    def __init__(self, num_control_points=4):
        super().__init__()
        self.num_control_points = num_control_points

        # Initialize control points near identity (linear mapping)
        self.control_points_1 = nn.Parameter(torch.tensor([0.0, 0.33, 0.66, 1.0]))
        self.control_points_2 = nn.Parameter(torch.tensor([0.0, 0.20, 0.80, 1.0]))
        self.control_points_3 = nn.Parameter(torch.tensor([0.0, 0.25, 0.75, 1.0]))

    def forward(self, x, visual=False):
        """
        Args:
            x: [C, H, W] input image with C in {1, 3}, values in [0, 1]
            visual: if True, also return a side-by-side visualization

        Returns:
            output: [3, H, W] transformed image
            combined_output: np.uint8 visualization array, or None
        """
        assert x.dim() == 3 and x.size(0) in (1, 3), "Input must be [C, H, W] with C in {1, 3}"
        c = x.size(0)

        cp1 = torch.sigmoid(self.control_points_1)
        cp2 = torch.sigmoid(self.control_points_2)
        cp3 = torch.sigmoid(self.control_points_3)

        if c == 1:
            f1 = self.bezier_curve(cp1, x)
            f2 = self.bezier_curve(cp2, x)
            f3 = self.bezier_curve(cp3, x)
            output = torch.cat([f1, f2, f3], dim=0)
        else:
            output = torch.cat([
                self.bezier_curve(cp1, x[0:1]),
                self.bezier_curve(cp2, x[1:2]),
                self.bezier_curve(cp3, x[2:3]),
            ], dim=0)

        combined_output = self._visualize(x, output) if visual else None
        return output, combined_output

    def bezier_curve(self, control_points, x):
        """
        Apply a cubic Bezier mapping to image intensities.

        Args:
            control_points: [4] tensor of y-coordinates (P0..P3)
            x: [1, H, W] image values in [0, 1]

        Returns:
            [1, H, W] remapped values in [0, 1]
        """
        P0, P1, P2, P3 = control_points
        t = x
        omt = 1 - t
        f_t = (comb(3, 0) * omt ** 3 * P0
               + comb(3, 1) * t * omt ** 2 * P1
               + comb(3, 2) * t ** 2 * omt * P2
               + comb(3, 3) * t ** 3 * P3)
        return torch.clamp(f_t, 0.0, 1.0)

    def _visualize(self, gray, out):
        """
        Build a side-by-side visualization of input and transformed channels.

        Args:
            gray: [C, H, W]
            out:  [3, H, W]

        Returns:
            np.uint8 [H, W*(N+2), 3]
        """
        def to_rgb(img2d):
            return np.stack([img2d] * 3, axis=-1)

        def norm(img):
            img = img.astype(np.float32)
            mn, mx = img.min(), img.max()
            img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)
            return (img * 255).astype(np.uint8)

        gray_np = gray.detach().cpu().numpy()
        base = to_rgb(norm(gray_np[0])) if gray_np.shape[0] == 1 else np.stack(
            [norm(c) for c in gray_np], axis=-1
        )
        canvases = [base]

        out_np = out.detach().cpu().numpy()
        for i in range(out_np.shape[0]):
            canvases.append(to_rgb(norm(out_np[i])))

        if out_np.shape[0] >= 3:
            rgb_comp = np.stack([norm(out_np[j]) for j in range(3)], axis=-1)
        else:
            last = out_np[-1]
            rgb_comp = np.stack([
                norm(out_np[j] if j < out_np.shape[0] else last) for j in range(3)
            ], axis=-1)
        canvases.append(rgb_comp)

        return np.concatenate(canvases, axis=1)


class RandomBezierTransform(nn.Module):
    """
    Non-learnable random Bezier intensity transform (useful for ablations).

    Samples random control points each forward pass for stochastic augmentation.

    Args:
        num_control_points: must be 4 (cubic Bezier)
        monotonic: enforce monotonic mapping to avoid color inversions
        strength: perturbation magnitude in [0, 1]
        share_across_rgb: use the same curve for all RGB channels
        seed: optional fixed random seed for reproducibility
    """

    def __init__(
        self,
        num_control_points: int = 4,
        monotonic: bool = True,
        strength: float = 0.25,
        mode: str = "medium",
        share_across_rgb: bool = False,
        seed=None,
    ):
        super().__init__()
        assert num_control_points == 4
        self.n = num_control_points
        self.monotonic = monotonic
        self.share = share_across_rgb
        if mode == "mild":
            strength = 0.15
        elif mode == "strong":
            strength = 0.40
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.base_cp = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
        self.fixed_rng = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            self.fixed_rng = g

    @torch.no_grad()
    def _sample_cp(self, device=None):
        """Sample one set of control points with optional monotonicity constraint."""
        noise = torch.empty(2, device=device).uniform_(-self.strength, self.strength, generator=self.fixed_rng)
        mid = self.base_cp[1:3].to(device) + noise
        if self.monotonic:
            mid = torch.sort(torch.clamp(mid, 0.0, 1.0)).values
        else:
            mid = mid.clamp(0.0, 1.0)
        return torch.tensor([0.0, mid[0].item(), mid[1].item(), 1.0], device=device)

    @staticmethod
    def _bezier_eval(cp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        P0, P1, P2, P3 = cp
        t = x
        omt = 1.0 - t
        y = (comb(3, 0) * omt ** 3 * P0
             + comb(3, 1) * t * omt ** 2 * P1
             + comb(3, 2) * t ** 2 * omt * P2
             + comb(3, 3) * t ** 3 * P3)
        return torch.clamp(y, 0.0, 1.0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, visual: bool = False):
        """
        Args:
            x: [C, H, W] with C in {1, 3}, values in [0, 1]
            visual: if True, return a visualization strip

        Returns:
            output: [3, H, W]
            viz: np.uint8 strip or None
        """
        assert x.dim() == 3 and x.size(0) in (1, 3)
        C = x.size(0)
        device = x.device

        if C == 1:
            outs = [self._bezier_eval(self._sample_cp(device), x) for _ in range(3)]
        elif self.share:
            cp = self._sample_cp(device)
            outs = [self._bezier_eval(cp, x[i:i + 1]) for i in range(3)]
        else:
            outs = [self._bezier_eval(self._sample_cp(device), x[i:i + 1]) for i in range(3)]

        out = torch.cat(outs, dim=0)

        if not visual:
            return out, None

        x_np = x.detach().cpu().numpy()
        out_np = out.detach().cpu().numpy()
        base = x_np[0] if C == 1 else np.stack([
            (x_np[i] - x_np[i].min()) / (np.ptp(x_np[i]) + 1e-6) if np.ptp(x_np[i]) > 0
            else np.zeros_like(x_np[i]) for i in range(3)
        ], axis=0).mean(0)

        def _to_rgb_strip(arr_list):
            imgs = []
            for a in arr_list:
                a = a.astype(np.float32)
                mn, mx = a.min(), a.max()
                a = (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a)
                imgs.append(np.stack([(a * 255).astype(np.uint8)] * 3, axis=-1))
            return np.concatenate(imgs, axis=1)

        strip = _to_rgb_strip([base, out_np[0], out_np[1], out_np[2]])
        return out, strip
