import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Model.protrain_model import block_Embedding



# class pretrain_data(Dataset):
#     def __init__(self):
#         self.All_data = []
#         Pretrain_path = 'F:/python/CHB-3/Train_data/Pretrain'
#         all_items = os.listdir(Pretrain_path)
#         for i in all_items:
#             data = np.load(Pretrain_path+'/'+i)
#             self.All_data.append(data)
#         self.All_data = np.array(self.All_data)
#
#     def __len__(self):
#         seq_len = self.All_data.shape[2]
#         return int(seq_len/512)
#
#     def __getitem__(self, item):
#         window_size = 512
#         start_pos = item * window_size
#         end_pos = (item + 1) * window_size
#         data_x = self.All_data[:, :, start_pos:end_pos]
#         data_x = torch.FloatTensor(data_x)
#         return data_x



class pretrain_data(Dataset):
    def __init__(self, data_path='F:/python/CHB-3/Train_data/Pretrain', window_size: int = 512, stride: int = 128):
        self.window_size = window_size
        self.stride = stride
        self.data_path = data_path
        # 检查路径是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"路径不存在: {data_path}")
        # 加载所有数据文件（按文件名排序）
        self.data_files = sorted([f for f in os.listdir(data_path) if f.endswith('.npy')])
        if len(self.data_files) != 8:
            raise ValueError(f"应包含8个.npy文件，但找到{len(self.data_files)}个")
        # 合并所有文件数据（沿时间维度拼接）
        self.full_data = []
        for file in self.data_files:
            data = np.load(os.path.join(data_path, file))
            if data.ndim != 2:
                raise ValueError(f"文件 {file} 应为二维 [channel, seq_len]，实际形状 {data.shape}")
            self.full_data.append(data)
        # 最终数据形状：[channel, total_seq_len]
        self.full_data = np.concatenate(self.full_data, axis=1)
        self.num_channels, self.total_length = self.full_data.shape
        # 预计算所有窗口的起止位置
        self.window_indices = [
            (i, i + window_size)
            for i in range(0, self.total_length - window_size + 1, stride)
        ]

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        start, end = self.window_indices[idx]
        window = self.full_data[:, start:end]  # 直接返回原始数据
        return torch.from_numpy(window).float()  # 转为torch张量但不做标准化

loader = DataLoader(dataset=pretrain_data(), batch_size=8, shuffle=True)

