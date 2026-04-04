import numpy as np
import torch
import torch.nn as nn
from torch import lgamma


class BezierPredictor2D(nn.Module):
    """Lightweight CNN that predicts Bezier control points from a grayscale image."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(16, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.encoder(x)


class DynamicBezierTransform2D(nn.Module):
    """
    Image-conditioned Bezier intensity transform.

    Predicts per-image Bezier control points from the input image content,
    then applies the resulting curves as a per-channel intensity remapping.

    Args:
        degree: Bezier polynomial degree (number of control points = degree + 1)
        num_curves: number of output channels (typically 3 for RGB)
    """

    def __init__(self, degree: int = 5, num_curves: int = 3):
        super().__init__()
        self.degree = degree
        self.num_cp = degree + 1
        self.tau = 5.0  # softmax temperature: higher = smoother control point distribution
        self.num_curves = num_curves

        self.predictor = BezierPredictor2D(out_dim=self.num_curves * self.num_cp)

    @staticmethod
    def bezier_curve(cp, t):
        """
        Evaluate a Bezier curve at positions t.

        Args:
            cp: [B, C, 1, 1, n+1] control point y-coordinates
            t:  [B, 1, H, W] pixel values in [0, 1]

        Returns:
            [B, C, H, W] transformed values in [0, 1]
        """
        n = cp.size(-1) - 1
        i = torch.arange(n + 1, device=t.device, dtype=t.dtype)

        log_fact = lgamma(i + 1)
        log_binom = log_fact[-1] - log_fact - log_fact.flip(0)
        binom = torch.exp(log_binom).view(1, 1, 1, 1, -1)

        t = t.unsqueeze(-1)
        basis = binom * (1 - t) ** (n - i) * t ** i
        f_t = (basis * cp).sum(dim=-1)
        return torch.clamp(f_t, 0.0, 1.0)

    def forward(self, x, visual=False):
        """
        Args:
            x: [C, H, W] input image with C in {1, 3}, values in [0, 1]
            visual: if True, also return a visualization numpy array

        Returns:
            output: [3, H, W] transformed image
            combined_output: np.uint8 [H, W*(N+2), 3] visualization, or None
        """
        assert x.dim() == 3 and x.size(0) in (1, 3), "Input must be [C, H, W] with C in {1, 3}"
        c, h, w = x.shape

        x_b = x.unsqueeze(0)
        gray = x_b.mean(dim=1, keepdim=True)

        # Predict and normalize control points
        cps = self.predictor(gray)
        cps = cps.view(1, self.num_curves, 1, 1, self.num_cp)
        cps = torch.softmax(cps / self.tau, dim=-1)

        cp1, cp2, cp3 = cps[:, 0], cps[:, 1], cps[:, 2]

        if c == 1:
            x1 = x.unsqueeze(0)
            output = torch.cat([
                self.bezier_curve(cp1, x1),
                self.bezier_curve(cp2, x1),
                self.bezier_curve(cp3, x1),
            ], dim=1).squeeze(0)
        else:
            output = torch.cat([
                self.bezier_curve(cp1, x[0:1].unsqueeze(0)),
                self.bezier_curve(cp2, x[1:2].unsqueeze(0)),
                self.bezier_curve(cp3, x[2:3].unsqueeze(0)),
            ], dim=1).squeeze(0)

        combined_output = self._visualize(gray, output) if visual else None
        return output, combined_output

    def _visualize(self, gray, out):
        """
        Build a side-by-side visualization of the input and Bezier-transformed channels.

        Args:
            gray: [1, 1, H, W] grayscale input
            out:  [3, H, W] transformed output

        Returns:
            np.uint8 array [H, W*(N+2), 3]
        """
        if gray.dim() == 4:
            gray = gray.squeeze(0)
        if out.dim() == 4:
            out = out.squeeze(0)

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
