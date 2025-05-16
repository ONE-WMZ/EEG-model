import torch
import torch.nn as nn
import math


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        b, t, c = x.size()
        # 平均池化分支
        avg_out = self.avg_pool(x.transpose(1, 2))  # (b, c, 1)
        avg_out = avg_out.view(b, c)
        avg_out = self.fc(avg_out)  # (b, c)
        # 最大池化分支
        max_out = self.max_pool(x.transpose(1, 2))  # (b, c, 1)
        max_out = max_out.view(b, c)
        max_out = self.fc(max_out)  # (b, c)
        # 合并注意力权重
        out = avg_out + max_out  # (b, c)
        return x * out.unsqueeze(1)  # (b, t, c)

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, output_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # 双向LSTM输出通道数为 hidden_dim*2
        self.attention = ChannelAttention(channel=hidden_dim * 2)
        # 调整全连接层输入维度
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # 初始化
        self._init_weights()
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        # 通道注意力
        attn_out = self.attention(lstm_out)  # (batch, seq_len, hidden_dim*2)
        # 取最后一个时间步
        last_step = attn_out[:, -1, :]  # (batch, hidden_dim*2)
        # 全连接层
        output = self.fc(last_step)  # (batch, output_dim)
        return output