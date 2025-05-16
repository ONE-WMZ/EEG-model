import torch
import torch.nn as nn
import numpy as np

"""  重要须知：(3)
                            3、模型：bi-lstm：（搭建）
                                    3-1、输入（batch_size, input_size, sqe_len）= (8, 23, 512)
                                    3-2、模型加载到 GPU 上
                                    3-3、提取正反向序列特征
                                    3-4、采用  掩码-重建  思想
                                    3-5、输出 时序特征 
"""

# ------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                                 # 加载到GPU上
# ------------------------------------------------------------------------------------------------------------------
def generate_mask(shape, mask_ratio=0.5):                                                              # 掩码函数
    mask = torch.ones(shape)
    mask[torch.rand(shape) < mask_ratio] = 0  # 随机掩码
    return mask

# ------------------------------------------------------------------------------------------------------------------

class bi_lstm(nn.Module):                                                                             # 模型：Bi-LSTM
    def __init__(self):
        super(bi_lstm, self).__init__()
        self.LSTM = nn.LSTM(23, 23, 2, bidirectional=True, batch_first=True)                          # LSTM
        self.fc = nn.Linear(23*2, 23)                                                      # Linear


    def forward(self, x):
        x = x.permute(0, 2, 1)                                            # 转换形状：(batch_size, seq_len, input_size)
        h0 = torch.zeros(2*2, 8, 23).to(device)                             # h0 = (num_lay*2, batch_size, hidden_size)
        c0 = torch.zeros(2*2, 8, 23).to(device)
        x = x.to(device)
        output, (hn, cn) = self.LSTM(x, (h0, c0))
        hn = hn.permute(1, 0, 2)
        print('hn.shape:', hn.shape)
        # hn = hn.reshape(8, 1)
        eeg_elmo = torch.cat((hn[-2], hn[-1]), dim=1)                                         # 特征拼接
        output = self.fc(output)
        return output, hn                                                                      # 返回 ：输出，特征

# model = bi_lstm().to(device)

# ------------------------------------------------------------------------------------------------------------------
# 测试代码：
# np.random.seed(42)                                                                                         # 随机种子
# data_1 = np.random.rand(8, 23, 512)                                                                        # 生成数据
# data_1 = torch.Tensor(data_1)
# data_1.to(device)
# mask = generate_mask(data_1.shape)
# data_1 = data_1*mask
# out, feature = model(data_1)
# # print(out.shape)
# # print(feature.shape)
