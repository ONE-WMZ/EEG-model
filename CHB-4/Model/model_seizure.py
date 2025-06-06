import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------------
""" 多尺度+注意力+残差 """
"""model_seizure_4 """
# class MultiScaleConv1D(nn.Module):
#     """多尺度1D卷积模块（修正版）"""
#
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleConv1D, self).__init__()
#         # 三个不同尺度的卷积核
#         self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
#         self.conv10 = nn.Conv1d(in_channels, out_channels, kernel_size=10, padding=5)
#         self.conv20 = nn.Conv1d(in_channels, out_channels, kernel_size=20, padding=10)
#
#         # 用于融合的特征权重
#         self.attention = nn.Sequential(
#             nn.Linear(3 * out_channels, out_channels),
#             nn.ReLU(),
#             nn.Linear(out_channels, 3),
#             nn.Softmax(dim=1)
#         )
#
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         # 多尺度特征提取
#         x1 = self.conv5(x)
#         x2 = self.conv10(x)
#         x3 = self.conv20(x)
#
#         # 确保所有输出尺寸一致（取最小长度）
#         min_len = min(x1.size(2), x2.size(2), x3.size(2))
#         x1 = x1[:, :, :min_len]
#         x2 = x2[:, :, :min_len]
#         x3 = x3[:, :, :min_len]
#
#         # 计算注意力权重
#         b, c, l = x1.shape
#         att_input = torch.cat([x1.mean(-1), x2.mean(-1), x3.mean(-1)], dim=1)
#         weights = self.attention(att_input).view(b, 3, 1, 1)
#
#         # 加权融合
#         combined = torch.stack([x1, x2, x3], dim=1)
#         out = (combined * weights).sum(dim=1)
#
#         # 批归一化和激活
#         out = self.bn(out)
#         out = self.relu(out)
#
#         return out
#
#
# class ResidualBlock(nn.Module):
#     """残差块"""
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = MultiScaleConv1D(in_channels, out_channels)
#         self.conv2 = MultiScaleConv1D(out_channels, out_channels)
#
#         # 下采样
#         self.downsample = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
#             nn.BatchNorm1d(out_channels)
#         ) if in_channels != out_channels or stride != 1 else None
#
#         self.dropout = nn.Dropout(0.3)
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.dropout(out)
#         out += identity
#
#         return F.relu(out)
#
#
# class EEG_CNN(nn.Module):
#     """癫痫检测模型（修正版）"""
#     def __init__(self, input_shape=(6, 400), num_classes=2):
#         super(EEG_CNN, self).__init__()
#         channels, length = input_shape
#
#         # 初始卷积层（添加ceil_mode=True）
#         self.init_conv = nn.Sequential(
#             nn.Conv1d(channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2, ceil_mode=True)
#         )
#
#         # 残差块
#         self.layer1 = self._make_layer(32, 64, 2)
#         self.layer2 = self._make_layer(64, 128, 2)
#         self.layer3 = self._make_layer(128, 256, 2)
#
#         # 自适应池化
#         self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
#
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )
#
#     def _make_layer(self, in_channels, out_channels, blocks):
#         layers = []
#         layers.append(ResidualBlock(in_channels, out_channels))
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # 输入形状: (batch, 6, 400)
#         x = self.init_conv(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#
#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#
#         return x

# ----------------------------------------------------------------------------------------------------------------------
""" 多尺度+注意力"""
"""model_seizure_5 """
class LightweightMultiScaleConv(nn.Module):
    """修正后的轻量级多尺度卷积模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 确保out_channels能被3整除
        assert out_channels % 3 == 0, "out_channels must be divisible by 3"

        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=7, padding=3)

        # 高效注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(),
            nn.Linear(out_channels // 4, 3),  # 注意这里的输出是3，对应三个分支
            nn.Softmax(dim=1)
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.Mish()  # Mish 激活函数

    def forward(self, x):
        # 多尺度特征提取
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)

        # 拼接特征 -> [B, 63, L]
        out = torch.cat([x1, x2, x3], dim=1)

        # 计算注意力权重 -> [B, 3]
        weights = self.attention(out)

        # 将输出切分为三部分 -> 每个 [B, 21, L]
        branch_channels = out.shape[1] // 3
        x1_, x2_, x3_ = torch.split(out, split_size_or_sections=branch_channels, dim=1)

        # 加权融合
        weighted = torch.cat([
            x1_ * weights[:, 0].view(-1, 1, 1),
            x2_ * weights[:, 1].view(-1, 1, 1),
            x3_ * weights[:, 2].view(-1, 1, 1)
        ], dim=1)

        return self.act(self.bn(weighted))


class EEGLightNet(nn.Module):
    """修正后的6通道专用轻量网络"""

    def __init__(self, input_shape=(6, 400), num_classes=2):
        super().__init__()

        # 输入处理层
        self.input_block = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=1),  # 通道混合
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # 多尺度特征提取（确保输出通道数是3的倍数）
        self.feature_extractor = nn.Sequential(
            LightweightMultiScaleConv(16, 63),  # 63=3×21
            nn.MaxPool1d(2),
            nn.Dropout1d(0.5),
            LightweightMultiScaleConv(63, 126),  # 126=3×42
            nn.MaxPool1d(2),
            nn.Dropout1d(0.5),
            LightweightMultiScaleConv(126, 252),  # 252=3×84
            nn.AdaptiveAvgPool1d(1)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(252, 64),
            nn.Dropout(0.6),
            nn.Mish(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.feature_extractor(x)
        return self.classifier(x)
