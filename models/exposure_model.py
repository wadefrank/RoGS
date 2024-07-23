import torch
import torch.nn as nn

# 这段代码定义了一个名为ExposureModel的神经网络模块，用于处理图像的曝光调整。
class ExposureModel(nn.Module):
    def __init__(self, num_camera=None):
        super().__init__()
        self.num_camera = num_camera
        # 初始化曝光参数a和b为零，分别用于每个相机
        self.exposure_a = nn.Parameter(torch.zeros(size=(num_camera, 1), dtype=torch.float32))  # (N, 3) axis angle
        self.exposure_b = nn.Parameter(torch.zeros(size=(num_camera, 1), dtype=torch.float32))

    def forward(self, idx, image):
        # 如果只有一个相机，不做曝光调整，直接返回图像
        if self.num_camera == 1:
            return image
        
        # 获取指定相机的曝光参数a和b
        exposure_a = self.exposure_a[idx]
        exposure_b = self.exposure_b[idx]
        
        # 调整图像曝光，公式为 image = exp(a) * image + b
        image = torch.exp(exposure_a) * image + exposure_b
        
        # 将调整后的图像像素值裁剪到[0, 1]范围
        image = image.clamp(0, 1)
        return image
