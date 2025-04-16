import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import comb  # 用于组合数计算
from PIL import Image

class LearnableBezierTransform(nn.Module):
    def __init__(self, num_control_points=4):
        """
        初始化 LearnableBezierTransform 模块。
        
        Args:
            num_control_points (int): 贝塞尔曲线的控制点数量，默认是4（用于三次贝塞尔曲线）。
        """
        super(LearnableBezierTransform, self).__init__()
        self.num_control_points = num_control_points
        
        # 初始化三组控制点，每组控制点对应一次三次贝塞尔曲线变换
        # 控制点的 y 坐标在 [0,1] 范围内，确保映射的连续性
        # 使用 sigmoid 激活确保控制点始终在 [0,1] 范围内
        # 下述初始值表示大致的线性映射，可根据需要进行微调
        self.control_points_1 = nn.Parameter(torch.tensor([
            0.0,
            0.33,
            0.66,
            1.0
        ], dtype=torch.float32))
        
        self.control_points_2 = nn.Parameter(torch.tensor([
            0.0,
            0.2,
            0.8,
            1.0
        ], dtype=torch.float32))

        self.control_points_3 = nn.Parameter(torch.tensor([
            0.0,
            0.25,
            0.75,
            1.0
        ], dtype=torch.float32))
        self.control_points_11 = nn.Parameter(torch.tensor([
            0.0,
            0.33,
            0.66,
            1.0
        ], dtype=torch.float32))
        
        self.control_points_22 = nn.Parameter(torch.tensor([
            0.0,
            0.2,
            0.8,
            1.0
        ], dtype=torch.float32))

        self.control_points_33 = nn.Parameter(torch.tensor([
            0.0,
            0.25,
            0.75,
            1.0
        ], dtype=torch.float32))
        
    def forward_1(self, x):
        """
        前向传播函数。

        输入:
            x (torch.Tensor): 输入图像，形状为 [1, H, W]，值范围 [0, 1]。
        
        输出:
            torch.Tensor: 拼接后的三通道图像，形状为 [3, H, W]。
        """
        # 确保输入形状正确
        assert x.dim() == 3 and x.size(0) == 1, "输入张量形状应为 [1, H, W]"
        
        # 使用 sigmoid 确保控制点在 [0,1] 范围内
        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]

        # 使用三条贝塞尔曲线分别对 x 进行变换
        f1 = self.bezier_curve(cp1, x)  # [1, H, W]
        f2 = self.bezier_curve(cp2, x)  # [1, H, W]
        f3 = self.bezier_curve(cp3, x)  # [1, H, W]
        
        # 将三个变换后的结果在通道维度上拼接，得到 [3, H, W]
        output = torch.cat([f1, f2, f3], dim=0)
        
        # 保存图像（横向拼接三个变换结果可视化）
        self.save_image(x[0], f1[0], f2[0], f3[0], 'transformed')
        
        return output  # [3, H, W]
    
    def forward(self, x, base_name):
        """
        前向传播函数。

        输入:
            x (torch.Tensor): 输入图像，形状为 [1, H, W] 或 [3, H, W]，值范围 [0, 1]。
        
        输出:
            torch.Tensor: 
                - 若输入为单通道 [1, H, W]，则输出为 [3, H, W] 的三通道图像；
                - 若输入为三通道 [3, H, W]，则输出也为 [3, H, W]（分别对每个通道变换）。
        """
        assert x.dim() == 3, "输入张量应为 3 维，形状为 [C, H, W]"
        c, h, w = x.shape
        assert c in [1, 3], "仅支持单通道 [1, H, W] 或三通道 [3, H, W] 输入"

        # 使用 sigmoid 确保控制点在 [0,1] 范围内
        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]

        if c == 1:
            # 原逻辑：单通道输入 -> 三条曲线 -> 拼接为三通道输出
            f1 = self.bezier_curve(cp1, x)  # [1, H, W]
            f2 = self.bezier_curve(cp2, x)  # [1, H, W]
            f3 = self.bezier_curve(cp3, x)  # [1, H, W]
            
            output = torch.cat([f1, f2, f3], dim=0)  # [3, H, W]

            # 可视化保存：原图 + 3 条通道变换
            # self.save_image_single_channel(x[0], f1[0], f2[0], f3[0], 'transformed_single')
            # self.save_image(x[0], f1[0], f2[0], f3[0], base_name)
            return output
        
        else:  # c == 3
            # 三通道输入：分别对三个通道使用不同的 Bezier 曲线
            # 例如：通道0 -> cp1, 通道1 -> cp2, 通道2 -> cp3
            out_c0 = self.bezier_curve(cp1, x[0:1])  # [1, H, W]
            out_c1 = self.bezier_curve(cp2, x[1:2])  # [1, H, W]
            out_c2 = self.bezier_curve(cp3, x[2:3])  # [1, H, W]
            
            output = torch.cat([out_c0, out_c1, out_c2], dim=0)  # [3, H, W]
            # self.save_image(x[0], out_c0[0], out_c1[0], out_c2[0], base_name)

            # 可视化保存：原图的三通道 + 变换后的三通道
            # self.save_image_three_channel(x, output, 'transformed_rgb')
            return output
    def forward_2(self, x):
        """
        前向传播函数。

        输入:
            x (torch.Tensor): 输入图像，形状为 [1, H, W] 或 [3, H, W]，值范围 [0, 1]。
        
        输出:
            torch.Tensor: 
                - 若输入为单通道 [1, H, W]，则输出为 [3, H, W] 的三通道图像；
                - 若输入为三通道 [3, H, W]，则输出也为 [3, H, W]（分别对每个通道变换）。
        """
        assert x.dim() == 3, "输入张量应为 3 维，形状为 [C, H, W]"
        c, h, w = x.shape
        assert c in [1, 3], "仅支持单通道 [1, H, W] 或三通道 [3, H, W] 输入"

        # 使用 sigmoid 确保控制点在 [0,1] 范围内
        cp1 = torch.sigmoid(self.control_points_1)  # [4]
        cp2 = torch.sigmoid(self.control_points_2)  # [4]
        cp3 = torch.sigmoid(self.control_points_3)  # [4]
        cp11 = torch.sigmoid(self.control_points_11)  # [4]
        cp22 = torch.sigmoid(self.control_points_22)  # [4]
        cp33 = torch.sigmoid(self.control_points_33)  # [4]

        if c == 1:
            # 原逻辑：单通道输入 -> 三条曲线 -> 拼接为三通道输出
            f1 = self.bezier_curve(cp1, x)  # [1, H, W]
            f2 = self.bezier_curve(cp2, x)  # [1, H, W]
            f3 = self.bezier_curve(cp3, x)  # [1, H, W]
            f11 = self.bezier_curve(cp11, x)  # [1, H, W]
            f22 = self.bezier_curve(cp22, x)  # [1, H, W]
            f33 = self.bezier_curve(cp33, x)  # [1, H, W]
            
            output = torch.cat([f1, f2, f3], dim=0)  # [3, H, W]
            output_2 = torch.cat([f11, f22, f33], dim=0)  # [3, H, W]

            # 可视化保存：原图 + 3 条通道变换
            # self.save_image_single_channel(x[0], f1[0], f2[0], f3[0], 'transformed_single')
            return output, output_2
        
        else:  # c == 3
            # 三通道输入：分别对三个通道使用不同的 Bezier 曲线
            # 例如：通道0 -> cp1, 通道1 -> cp2, 通道2 -> cp3
            out_c0 = self.bezier_curve(cp1, x[0:1])  # [1, H, W]
            out_c1 = self.bezier_curve(cp2, x[1:2])  # [1, H, W]
            out_c2 = self.bezier_curve(cp3, x[2:3])  # [1, H, W]
            f11 = self.bezier_curve(cp11, x[0:1])  # [1, H, W]
            f22 = self.bezier_curve(cp22, x[1:2])  # [1, H, W]
            f33 = self.bezier_curve(cp33, x[2:3])  # [1, H, W]
            
            
            output = torch.cat([out_c0, out_c1, out_c2], dim=0)  # [3, H, W]
            output_2 = torch.cat([f11, f22, f33], dim=0)  # [3, H, W]

            # 可视化保存：原图的三通道 + 变换后的三通道
            # self.save_image_three_channel(x, output, 'transformed_rgb')
            return output, output_2
    
    
    def bezier_curve(self, control_points, x):
        """
        计算贝塞尔曲线的非线性变换。

        Args:
            control_points (torch.Tensor): 形状为 [4] 的一维控制点 (P0, P1, P2, P3)。
            x (torch.Tensor): 输入图像，形状为 [1, H, W]，值范围 [0,1]。
        
        Returns:
            torch.Tensor: 变换后的图像，形状为 [1, H, W]。
        """
        # 控制点 (P0, P1, P2, P3) 都是标量，表示 y 坐标
        P0, P1, P2, P3 = control_points
        
        # 定义 t 和 (1 - t)
        t = x                 # [1, H, W]
        one_minus_t = 1 - t   # [1, H, W]
        
        # 三次贝塞尔权重
        B0 = comb(3, 0) * (one_minus_t ** 3)      # [1, H, W]
        B1 = comb(3, 1) * (t ** 1) * (one_minus_t ** 2)
        B2 = comb(3, 2) * (t ** 2) * (one_minus_t ** 1)
        B3 = comb(3, 3) * (t ** 3) * (one_minus_t ** 0)
        
        # 计算像素映射值
        f_t = B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3
        
        # 限制在 [0,1] 范围
        f_t = torch.clamp(f_t, 0.0, 1.0)
        
        return f_t  # [1, H, W]
    
    def save_image(self, img, img1, img2, img3, filename):
        """
        保存 3 张灰度图为横向拼接的长图。

        Args:
            img1, img2, img3 (torch.Tensor): 分别为 [H, W] 的灰度图。
            filename (str): 保存文件名（不包含后缀 .png）。
        """
        # 转为 CPU 上的 NumPy 数组
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        img3_np = img3.detach().cpu().numpy()
        img_np = img.detach().cpu().numpy()

        # 规范化到 [0,255]
        img1_np = self.normalize_image(img1_np)
        img2_np = self.normalize_image(img2_np)
        img3_np = self.normalize_image(img3_np)
        img_np = self.normalize_image(img_np)

        # 横向拼接 => [H, 3*W]
        combined = np.concatenate([img_np, img1_np, img2_np, img3_np], axis=1)

        # 转为单通道 PIL 图像
        image = Image.fromarray(combined, mode='L')
        image.save(f"/home/data/SAM/wesam/show_4img/{filename}.png")
        print(f"Image saved as '{filename}.png'")

    @staticmethod
    def normalize_image(img):
        """
        将图像规范化到 [0,255] 并转换为 uint8。

        Args:
            img (np.ndarray): 输入图像 (H, W)。
        
        Returns:
            np.ndarray: 规范化并转换后的图像 (H, W)，dtype=uint8。
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

    # 应用三次独立变换，得到 [3, H, W]
    transformed_image = transform(input_image)
    print(f"Transformed image shape: {transformed_image.shape}, device: {transformed_image.device}")

    # 这里不用再手动 split，因为在 forward 里已经调用了 save_image。
    # 如果需要单独查看 3 个变换后的图像，可自行 split:
    # f1 = transformed_image[0].cpu().numpy()
    # f2 = transformed_image[1].cpu().numpy()
    # f3 = transformed_image[2].cpu().numpy()
