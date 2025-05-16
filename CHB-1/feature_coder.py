import torch
import torch.nn as nn
import numpy as np
"""  重要须知：(4)
                            3、coder_feature:
                                    3-1、输入（batch_size, input_size, sqe_len）= (8, 23, 512)
                                    3-2、Encoder: 512*23 ——> 128
                                    3-3、Decoder:  128 —— 512*23
                                    3-4、经过编码-解码直接计算生成后的序列和原序列的loss
                                    3-5、采用Mask_restruct计算loss
                                    3-6、LSTM层数：10
"""


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):                                                        # 编码器
    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(23, 256, 10, bidirectional=True, batch_first=True)                 # LSTM
        self.fc = nn.Linear(256, 128)                                     # feature_size = 128

    def forward(self, x):
        h0 = torch.zeros(2 * 10, 8, 256).to(device)      # (num_lay*2, batch_size, hidden_size)
        c0 = torch.zeros(2 * 10, 8, 256).to(device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        hidden = hn[-1]
        feature = self.fc(hidden)
        return feature


class Decoder(nn.Module):                                                        # 解码器
    def __init__(self):
        super(Decoder, self).__init__()
        self.seq_len = 512
        self.lstm = nn.LSTM(23, 256, batch_first=True)
        self.fc = nn.Linear(256, 23)
        self.hidden_fc = nn.Linear(128, 256)

    def forward(self, feature):
        batch_size = feature.size(0)
        hidden = self.hidden_fc(feature).unsqueeze(0).to(device)  # [1, batch_size, hidden_size]
        cell = torch.zeros_like(hidden).to(device)  # 初始细胞状态
        # 初始输入（全零）
        x = torch.zeros(8, 1, self.fc.out_features) .to(device)   # [batch_size, 1, output_size]
        outputs = []
        for t in range(self.seq_len):
            output, (hidden, cell) = self.lstm(x, (hidden, cell))  # 解码一个时间步
            output = self.fc(output)  # 映射到输出空间
            outputs.append(output)
            x = output  # 使用当前输出作为下一个时间步的输入
        return torch.cat(outputs, dim=1)  # [batch_size, seq_length, output_size]

# ----------------------------------------------------------------------------------------------------------------
# 测试代码：
"""
encoder = Encoder()                                                                                        # 初始化
decoder = Decoder()
np.random.seed(42)                                                                                         # 随机种子
data_1 = np.random.rand(8, 512, 23)                                                                        # 生成数据
data_1 = torch.Tensor(data_1)
print(data_1.shape)
vc = encoder(data_1)                                            # 编码
print(vc.shape)
out = decoder(vc)                                               # 解码
print(out.shape)
loss_fn = nn.MSELoss()                                          # 损失
loss_ = loss_fn(data_1, out)
print('loss: ', loss_)
    输出：
        torch.Size([8, 512, 23])     # 原始数据
        torch.Size([8, 128])         # 特征向量
        torch.Size([8, 512, 23])     # 反向推出原始特征
        loss:  tensor(0.3413, grad_fn=<MseLossBackward0>)    # 计算出的损失
"""