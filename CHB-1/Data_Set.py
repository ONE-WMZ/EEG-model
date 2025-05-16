from Data_Process import get_address
import Data_Process
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


"""  重要须知：(2)
                            2、数据集--封装 ：
                                    2-1、数据分割（2s，不重叠）
                                    2-2、无标签数据
                                    2-3、文件分块（可根据内存大小调节块的大小）682个
                                    2-4、训练集和测试集都可以调用这个类
                                    2-5、参数: | block_size |  file_id  |
                                            |   块大小    |    文件    |
                                    2-6、data.item.shape = ([8, 23, 512])
                                      (batch_size, channels , sqe_len)
"""


address, _ = get_address()                                                                # 取地址
# ------------------------------------------------------------------------------------------------------------
class data_set(Dataset):                                                                  # 数据集-封装
    def __init__(self, file_id, block_size):
        self.window_size = 512                                                                      # 2s的滑窗
        self.data = []                                                                              # 数据缓存
        for i in range(file_id, file_id+block_size):                                                # 数量
            data_pro = Data_Process.data_preprocess(address[i])                                     # 数据预处理
            if i == file_id:
                self.data = data_pro
            else:
                self.data = np.hstack((self.data, data_pro))                                       # 按列添加

    def __len__(self):
        data_len = (self.data.shape[1] / self.window_size)    # 默认每个文件都是1个小时
        return int(data_len)

    def __getitem__(self, idx):
        start_idx = idx * self.window_size
        end_idx = (idx + 1) * self.window_size
        item = self.data[:, start_idx:end_idx]
        return torch.FloatTensor(item)
# -----------------------------------------------------------------------------------------------------

# 测试代码：
# data_set = data_set(1, 2)
# data_load = DataLoader(data_set, batch_size=8)
# print(len(data_load))
# for item in data_load:
#     print(item.shape)
#     break
