import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPHead(nn.Module):
    def __init__(self, in_channels_list, embed_dim=256, num_classes=5):
        """
        in_channels_list: ConvNeXt 4个stage的输出通道数，例如 [96, 192, 384, 768] (Tiny的默认配置)
        embed_dim: 统一降维后的通道数
        """
        super().__init__()
        
        # 为4个尺度的特征分别准备一个线性映射层，统一通道数到 embed_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for in_channels in in_channels_list
        ])
        
        # 融合后的最终分类层
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        # features 是一个包含4个张量的列表: [c1, c2, c3, c4]
        outs = []
        for i, f in enumerate(features):
            x = self.mlps[i](f) # 统一通道数
            # 将所有特征图上采样到第一个特征图（分辨率最高，通常是原图的 1/4）的大小
            x = F.interpolate(x, size=features[0].shape[2:], mode='bilinear', align_corners=False)
            outs.append(x)
            
        # 在通道维度拼接: 4 * embed_dim
        out = torch.cat(outs, dim=1) 
        out = self.linear_fuse(out)
        out = self.classifier(out)
        return out