import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidPoolingModule(nn.Module):
    """PPM 模块：通过不同尺度的池化获取全局上下文"""
    def __init__(self, in_channels, out_channels, bin_sizes=[1, 2, 4, 8]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for bin_size in bin_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(bin_sizes) * out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        out  = [x]
        for stage in self.stages:
            pooled = stage(x)
            out.append(F.interpolate(pooled, size=size, mode='bilinear', align_corners=False))
        out = torch.cat(out, dim=1)
        return self.bottleneck(out)

class UPerHead(nn.Module):
    def __init__(self, in_channels_list, pool_scales=[1, 2, 4, 8], channels=512, num_classes=7):
        super().__init__()
        # 最顶层特征经过 PPM 模块
        self.ppm = PyramidPoolingModule(in_channels_list[-1], channels, pool_scales)
        
        # FPN 侧边连接：将较低维度的特征映射到统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels_list[i], channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for i in range(len(in_channels_list) - 1)
        ])
        
        # FPN 融合后的卷积
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list) - 1)
        ])
        
        # 最终的 FPN 特征融合与分类层
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features):
        # 1. 最深层特征通过 PPM
        ppm_out = self.ppm(features[-1])
        fpn_outs = [ppm_out]
        
        # 2. 自顶向下的 FPN 融合
        for i in range(len(features) - 2, -1, -1):
            lateral_out = self.lateral_convs[i](features[i])
            top_down_out = F.interpolate(fpn_outs[-1], size=lateral_out.shape[2:], mode='bilinear', align_corners=False)
            fpn_out = self.fpn_convs[i](lateral_out + top_down_out)
            fpn_outs.append(fpn_out)
            
        fpn_outs = fpn_outs[::-1] # 翻转回原始层级顺序 [c1, c2, c3, c4]
        
        # 3. 将所有 FPN 层的输出上采样到 C1 的分辨率并拼接
        outs = [fpn_outs[0]]
        for i in range(1, len(fpn_outs)):
            outs.append(F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False))
            
        out = self.fpn_bottleneck(torch.cat(outs, dim=1))
        out = self.classifier(out)
        return out