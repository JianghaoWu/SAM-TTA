import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import comb  # 用于组合数计算
from PIL import Image

class LearnableBezierTransform(nn.Module):
    def __init__(self, num_control_points=4):
        """
        初始化LearnableBezierTransform模块。
        
        Args:
            num_control_points (int): 贝塞尔曲线的控制点数量，默认是4（用于三次贝塞尔曲线）。
        """
        super(LearnableBezierTransform, self).__init__()
        self.num_control_points = num_control_points
        
        # 初始化两个变换的控制点，每个变换有四个控制点（用于三次贝塞尔曲线）
        # 控制点的坐标在[0,1]范围内，确保映射的连续性
        # 使用sigmoid激活确保控制点始终在[0,1]范围内
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
        
        # 应用sigmoid确保控制点在[0,1]范围内
        cp1 = torch.sigmoid(self.control_points_1)  # [4,2]
        cp2 = torch.sigmoid(self.control_points_2)  # [4,2]
        
        # 生成贝塞尔曲线映射函数
        f1 = self.bezier_curve(cp1, x)  # [1, H, W]
        f2 = self.bezier_curve(cp2, x)  # [1, H, W]
        
        # 拼接原始图像和两个变换后的图像
        output = torch.cat([x, f1, f2], dim=0)  # [3, H, W]
        self.save_image(x[0], f1[0], f2[0], 'filename')
        
        return output  # [3, H, W]
    
    def bezier_curve(self, control_points, x):
        """
        计算贝塞尔曲线的非线性变换。
        
        Args:
            control_points (torch.Tensor): 控制点，形状为 [4,2]。
            x (torch.Tensor): 输入图像，形状为 [1, H, W]，值范围 [0,1]。
        
        Returns:
            torch.Tensor: 变换后的图像，形状为 [1, H, W]。
        """
        # 控制点坐标
        P0, P1, P2, P3 = control_points  # 每个是 [2]
        
        # 计算贝塞尔曲线的权重
        # 三次贝塞尔曲线权重公式: B_i(t) = C(3,i) * t^i * (1-t)^(3-i)
        # 其中i = 0,1,2,3
        
        # 展开公式
        t = x  # [1, H, W]
        one_minus_t = 1 - t  # [1, H, W]
        
        # 计算每个控制点的权重
        B0 = comb(3, 0) * (one_minus_t ** 3)  # [1, H, W]
        B1 = comb(3, 1) * (t ** 1) * (one_minus_t ** 2)  # [1, H, W]
        B2 = comb(3, 2) * (t ** 2) * (one_minus_t ** 1)  # [1, H, W]
        B3 = comb(3, 3) * (t ** 3) * (one_minus_t ** 0)  # [1, H, W]
        
        # 计算变换后的像素值
        # f(t) = B0 * P0_y + B1 * P1_y + B2 * P2_y + B3 * P3_y
        # 只使用y坐标来映射灰度值
        f_t = B0 * P0[1] + B1 * P1[1] + B2 * P2[1] + B3 * P3[1]  # [1, H, W]
        
        # 确保输出在[0,1]范围内
        f_t = torch.clamp(f_t, 0.0, 1.0)
        
        return f_t  # [1, H, W]
    
    def save_image(self, original, transformed1, transformed2, filename):
        """
        保存原图、变换后的两个通道为一个横向拼接的长图。
        
        Args:
            original (torch.Tensor): 原始图像，形状为 [H, W]。
            transformed1 (torch.Tensor): 第一个变换后的图像，形状为 [H, W]。
            transformed2 (torch.Tensor): 第二个变换后的图像，形状为 [H, W]。
            filename (str): 保存的文件名。
        """
        # 将 Tensor 转为 CPU 并转换为 NumPy 数组
        original_np = original.detach().cpu().numpy()
        transformed1_np = transformed1.detach().cpu().numpy()
        transformed2_np = transformed2.detach().cpu().numpy()

        # 规范化图像到 [0,255]
        original_np = self.normalize_image(original_np)
        transformed1_np = self.normalize_image(transformed1_np)
        transformed2_np = self.normalize_image(transformed2_np)

        # 水平方向拼接，生成形状为 [H, 3*W]
        combined = np.concatenate([original_np, transformed1_np, transformed2_np], axis=1)  # [H, 3W]

        # 转换为 PIL 图像并保存
        # 确保图像为单通道模式 'L'
        image = Image.fromarray(combined, mode='L')
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
    transform = LearnableBezierTransform().to(device)

    # 创建示例输入图像张量 [1, H, W]，值范围 [0,1]
    input_image = torch.rand(1, height, width, device=device)  # [1, 1024, 1024]

    # 应用变换
    transformed_image = transform(input_image)  # [3, 1024, 1024]

    # 打印输出信息
    print(f"Transformed image shape: {transformed_image.shape}, device: {transformed_image.device}")

    # 保存图像（仅保存拼接后的图像）
    # 将拼接后的图像分离
    original = transformed_image[0].detach()
    transformed1 = transformed_image[1].detach()
    transformed2 = transformed_image[2].detach()
    transform.save_image(original, transformed1, transformed2, 'transformed')
