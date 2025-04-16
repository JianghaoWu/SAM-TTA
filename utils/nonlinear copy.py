import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import comb  # 用于组合数计算
from PIL import Image

class LearnableBezierTransform(nn.Module):
    def __init__(self, height, width):
        """
        初始化LearnableBezierTransform模块。
        
        Args:
            height (int): 图像高度。
            width (int): 图像宽度。
        """
        super(LearnableBezierTransform, self).__init__()
        self.height = height
        self.width = width

        # 初始化两个变换的控制点，每个变换有四个控制点（用于三次贝塞尔曲线）
        # 控制点的x和y坐标在[0,1]范围内，确保映射的连续性
        # 初始控制点设为线性映射（身份变换）
        self.control_points_1 = nn.Parameter(torch.tensor([
            [0.0, 0.0],
            [0.33, 0.0],
            [0.66, 1.0],
            [1.0, 1.0]
        ], dtype=torch.float32))
        
        self.control_points_2 = nn.Parameter(torch.tensor([
            [0.0, 0.0],
            [0.33, 0.0],
            [0.66, 1.0],
            [1.0, 1.0]
        ], dtype=torch.float32))

    def forward(self, x):
        """
        前向传播函数。
        
        Args:
            x (torch.Tensor): 输入图像，形状为 [1, H, W]，值范围 [0,1]。
        
        Returns:
            torch.Tensor: 拼接后的三通道图像，形状为 [3, H, W]。
        """
        # 确保输入形状为 [1, H, W]
        assert x.dim() == 3 and x.size(0) == 1, "输入张量形状应为 [1, H, W]"
        
        device = x.device

        # 生成第一个贝塞尔曲线映射
        grid1 = self.create_grid(self.control_points_1, device)

        # 生成第二个贝塞尔曲线映射
        grid2 = self.create_grid(self.control_points_2, device)

        # 调整输入形状以适应 grid_sample，[1, 1, H, W]
        x = x.unsqueeze(0)  # [1,1,H,W]

        # 使用 grid_sample 进行可微分的变换
        transformed1 = F.grid_sample(x, grid1, mode='bilinear', padding_mode='border', align_corners=True)  # [1,1,H,W]
        transformed2 = F.grid_sample(x, grid2, mode='bilinear', padding_mode='border', align_corners=True)  # [1,1,H,W]

        # 去掉多余的维度
        transformed1 = transformed1.squeeze(0).squeeze(0)  # [H,W]
        transformed2 = transformed2.squeeze(0).squeeze(0)  # [H,W]

        # 拼接原始图像和变换后的通道
        output = torch.stack([x.squeeze(0).squeeze(0), transformed1, transformed2], dim=0)  # [3,H,W]
        # print(x.shape, transformed1.shape, transformed2.shape, output.shape)
        self.save_image(x[0][0], transformed1, transformed2, output, 'transformed')


        return output

    def bezier_curve(self, control_points, t):
        """
        计算贝塞尔曲线。
        
        Args:
            control_points (torch.Tensor): 控制点，形状为 [4,2]。
            t (torch.Tensor): 参数，形状为 [N]。
        
        Returns:
            torch.Tensor: 贝塞尔曲线上的点，形状为 [N,2]。
        """
        nPoints = control_points.size(0)  # 应该是4
        binomial = torch.tensor([comb(nPoints - 1, i) for i in range(nPoints)], dtype=torch.float32, device=t.device).view(1, nPoints)  # [1,4]

        t = t.view(-1, 1)  # [N,1]
        exponents = torch.arange(nPoints, device=t.device).float().view(1, nPoints)  # [1,4]

        # 计算 (t^i) * (1-t)^(nPoints-1-i) for i in [0,3]
        basis = binomial * (t ** exponents) * ((1 - t) ** (nPoints - 1 - exponents))  # [N,4]

        # 计算贝塞尔曲线上的点
        curve = torch.matmul(basis, control_points)  # [N,2]

        return curve  # [N,2]

    def create_grid(self, control_points, device):
        """
        创建用于 grid_sample 的网格。
        
        Args:
            control_points (torch.Tensor): 控制点，形状为 [4,2]。
            device (torch.device): 设备。
        
        Returns:
            torch.Tensor: 网格，形状为 [1,H,W,2]，范围在 [-1,1]。
        """
        # 生成t参数
        t_x = torch.linspace(0, 1, steps=self.width, device=device)  # [W]
        t_y = torch.linspace(0, 1, steps=self.height, device=device)  # [H]

        # 计算f(x)和g(y)通过贝塞尔曲线
        f_x = self.bezier_curve(control_points, t_x)[:, 1]  # [W]
        g_y = self.bezier_curve(control_points, t_y)[:, 1]  # [H]

        # 创建二维网格
        grid_x = f_x.unsqueeze(0).repeat(self.height, 1)  # [H,W]
        grid_y = g_y.unsqueeze(1).repeat(1, self.width)  # [H,W]

        # 合并为 [H,W,2]
        grid = torch.stack([grid_x, grid_y], dim=2)  # [H,W,2]

        # 规范化到 [-1,1] 范围
        grid = grid * 2.0 - 1.0  # 假设 f_x 和 g_y 在 [0,1] 范围内

        # 添加批次维度
        grid = grid.unsqueeze(0)  # [1,H,W,2]

        # 确保 grid 在正确的设备和类型
        grid = grid.to(device).float()

        return grid  # [1,H,W,2]

    def save_image(self, original, channel1, channel2, transformed, filename):
        """
        保存原图、变换后的两个通道和拼接结果为本地图像。
        
        Args:
            original (torch.Tensor): 原始图像，形状为 [H,W]。
            channel1 (torch.Tensor): 第一个变换后的图像，形状为 [H,W]。
            channel2 (torch.Tensor): 第二个变换后的图像，形状为 [H,W]。
            transformed (torch.Tensor): 拼接后的图像，形状为 [3,H,W]。
            filename (str): 保存的文件名。
        """
        # 将 Tensor 转为 CPU 并转换为 NumPy 数组
        original_np = original.detach().cpu().numpy()
        channel1_np = channel1.detach().cpu().numpy()
        channel2_np = channel2.detach().cpu().numpy()

        # 规范化图像到 [0,255]
        original_np = self.normalize_image(original_np)
        channel1_np = self.normalize_image(channel1_np)
        channel2_np = self.normalize_image(channel2_np)

        # 水平方向拼接
        combined = np.concatenate([original_np, channel1_np, channel2_np], axis=1)  # [H, 3W]

        # 转换为 PIL 图像并保存
        image = Image.fromarray(combined)
        image.save(f"{filename}.png")
        print(f"Image saved as '{filename}.png'")

    @staticmethod
    def normalize_image(img):
        """
        规范化图像到 [0,255] 并转换为 uint8。
        
        Args:
            img (np.ndarray): 输入图像。
        
        Returns:
            np.ndarray: 规范化后的图像。
        """
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        return (img * 255).astype(np.uint8)

# 测试示例
if __name__ == "__main__":
    # 定义图像尺寸
    height, width = 1024, 1024

    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化变换模块并移动到设备
    transform = LearnableBezierTransform(height, width).to(device)

    # 创建示例输入图像张量 [1, H, W]，值范围 [0,1]
    input_image = torch.rand(1, height, width, device=device)  # [1, 1024, 1024]

    # 应用变换
    with torch.no_grad():  # 禁用梯度计算（仅用于测试）
        transformed_image = transform(input_image)

    # 打印输出信息
    print(f"Transformed image shape: {transformed_image.shape}, device: {transformed_image.device}")

    # 如果需要保存图像，可以取消注释以下行
    # transform.save_image(input_image.squeeze(0), transformed1, transformed2, transformed_image, 'transformed')
