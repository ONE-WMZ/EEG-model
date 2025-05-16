import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
        文件说明：
                采用多任务方式进行特征泛化
                多任务共用一个编码器进行特征提取，参数共享
                （其他思路：多任务分别用一个编码器，在训练期间，随机时间进行编码器相似操作，找最优编码器）
                
"""
# ———————————————————————————————————————————      时间特征提取模块(1D-CNN)    ————————————————————————————————————————————
# class Time_feature_block(nn.Module):
#     """
#                         输入形状: (time_block, channels, seq_len)      = (8, 22, 64)
#                         输出形状：(time_block, channels, time_feature) = (8, 22, 32)
#     """
#     def __init__(self,):   # input_size =22*64, output_size=22*32
#         super(Time_feature_block, self).__init__()
#         # 输入形状: (num_block, channels, length) = (8, 22, 64)
#         # 第一层卷积：保持时间维度不变
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=22, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU()
#         )
#         # 第二层卷积：下采样
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU()
#         )
#         # 输出层：调整通道数
#         self.final_conv = nn.Conv1d(32, 22, kernel_size=1)
#
#     def forward(self, x):
#         x = self.conv1(x)                # [8, 64, 64]
#         x = self.conv2(x)                # [8, 32, 32]
#         x = self.final_conv(x)           # [8, 22, 32]
#         return x
class Time_feature_block(nn.Module):
    """
    改进说明：
    1. 网络深度增加至4层（原2层）
    2. 引入残差连接（Residual Blocks）
    3. 保持输入输出形状不变：(8, 22, 64) → (8, 22, 32)
    """
    def __init__(self):
        super(Time_feature_block, self).__init__()
        # Block 1: [8, 22, 64] -> [8, 64, 64] (保持长度)
        self.block1 = nn.Sequential(
            nn.Conv1d(22, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # Block 2: [8, 64, 64] -> [8, 64, 32] (下采样)
        self.downsample1 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        # Block 3: [8, 64, 32] -> [8, 128, 32] (扩展通道)
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # Block 4: [8, 128, 32] -> [8, 32, 16] (下采样)
        self.downsample2 = nn.Conv1d(128, 32, kernel_size=3, stride=2, padding=1)
        # 输出层调整
        self.final_conv = nn.Conv1d(32, 22, kernel_size=1)
        # 残差路径的适配层
        self.residual_conv1 = nn.Conv1d(22, 64, kernel_size=1)
        self.residual_conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.residual_conv3 = nn.Conv1d(128, 32, kernel_size=1)
    def forward(self, x):
        identity = x
        # 第一层处理
        x = self.block1(x)  # [8, 64, 64]
        identity = self.residual_conv1(identity)  # [8, 64, 64]
        x = x + identity
        x = F.relu(x)
        # 第一次下采样
        x = self.downsample1(x)  # [8, 64, 32]
        identity = F.avg_pool1d(identity, kernel_size=2)  # [8, 64, 32]
        identity = self.residual_conv2(identity)  # [8, 128, 32]
        x = self.block2(x)  # [8, 128, 32]
        x = x + identity
        x = F.relu(x)
        # 第二次下采样
        x = self.downsample2(x)  # [8, 32, 16]
        identity = F.avg_pool1d(identity, kernel_size=2)  # [8, 128, 16]
        identity = self.residual_conv3(identity)  # [8, 32, 16]
        x = x + identity
        x = F.relu(x)
        # 恢复时间维度到32
        x = F.interpolate(x, size=32, mode='linear', align_corners=True)  # [8, 32, 32]
        # 最终通道调整
        return self.final_conv(x)  # [8, 22, 32]

# ———————————————————————————————————————————      自注意力模块    ———————————————————————————————————————————————————————
class Self_attention_block(nn.Module):
    # 时间块之间进行自注意力
    """
                    输入形状: (batch_size, time_block, time_feature) = (8, 8, 32)
                    输出形状：(batch_size, fuse_block, fuse_feature) = (8, 1, 64)
    """
    def __init__(self, input_dim=32, output_dim=64):
        super(Self_attention_block, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
        # 如果仍有问题，强制所有参数到设备（极端情况用）
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        # 输入形状 x: (batch_size, time_block, input_dim) = (8, 8, 32)
        x = x.to(device)
        assert x.device.type == 'cuda', f"Input x is on {x.device}, expected CUDA"    # 设备检查
        Q = self.query(x)  # (8, 8, 64)
        K = self.key(x)  # (8, 8, 64)
        V = self.value(x)  # (8, 8, 64)
        # 计算注意力权重 (8, 8, 8)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (64 ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # 加权求和 (8, 8, 64)
        weighted_values = torch.bmm(attention_weights, V)
        # 压缩序列维度到 1（取均值或注意力池化）
        pooled_output = weighted_values.mean(dim=1, keepdim=True)  # (8, 1, 64)
        # 最终输出变换
        output = self.output_proj(pooled_output)
        assert output.device.type == 'cuda', f"Output is on {output.device}, expected CUDA"
        return output   # (8, 1, 64)

# ———————————————————————————————————————————      通道注意力模块（CBAM）    ——————————————————————————————————————————————
class Channel_attention(nn.Module):
    # 通道之间计算注意力
    """
                输入形状: (batch_size, channels, time_feature)   = (8, 22, 32)
                输出形状：(batch_size, fuse_chan, time_feature)  = (8, 1,32)
    """
    def __init__(self, channel, ratio=2):
        super(Channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, l = [8, 22, 32] ->  [8, 1, 32]
        time_block, channel_num, seq_len = x.size()
        avg_pool = self.avg_pool(x).view([time_block, channel_num])
        max_pool = self.max_pool(x).view([time_block, channel_num])
        avg_fc = self.fc(avg_pool)
        max_fc = self.fc(max_pool)
        out = avg_fc + max_fc
        out = self.sigmoid(out).view([time_block, channel_num, 1])
        fused = (x * out).sum(dim=1, keepdim=True)
        return fused   # [8, 1, 32]

# ———————————————————————————————————————————      编码器    ————————————————————————————————————————————————————————————
class Encode(nn.Module):
    """
            输入形状: (batch_size, channels, seq_len) = (8, 22, 512)
            输出形状：(batch_size, feature_size) = (8, 64)
    """
    def __init__(self):
        super(Encode, self).__init__()
        self.one_block = Time_feature_block()
        self.two_block = Channel_attention(22)   # channel=22 , ratio=2
        self.three_block = Self_attention_block()

    def forward(self, x):
        x = x.view(8, 22, 8, 64)  # (8, 22, 512)   -> (8, 22, 8, 64)
        x = x.transpose(1, 2)     # (8, 22, 8, 64) -> (8, 8, 22, 64)
        outputs = torch.empty((8, 8, 32))
        for i in range(x.shape[0]):
            xi = x[i]                              # (8,22,64)
            time_i = self.one_block(xi)            # (8,22,64) -> (8,22,32)
            channel_i = self.two_block(time_i)     # (8,22,32) -> (8,1,32)
            x_reshaped = channel_i.view(8, 32)     # (8,1,32)  -> (8,32)
            outputs[i] = x_reshaped                # 8个(8,32) ->(8,8,32)
        out = self.three_block(outputs)            # (8,8,32) -> (8,1,64)
        out = out.view(8, 64)                      # (8,1,64) -> (8,64)
        return out                                 # (8,64)


