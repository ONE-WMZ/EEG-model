import os
import torch
import numpy as np
from Model.protrain_model import block_Embedding
from torch.utils.data import Dataset, DataLoader


class Down_train_data(Dataset):
    def __init__(self):
        self.data = []  # 存储所有数据样本
        self.labels = []  # 存储对应标签
        pretrain_path = 'F:/python/CHB-3/Train_data/Downstream'
        # 遍历类别文件夹（0和1）
        for class_name in os.listdir(pretrain_path):
            class_path = os.path.join(pretrain_path, class_name)
            # 确保是文件夹
            if not os.path.isdir(class_path):
                continue
            # 获取类别标签（0或1）
            label = int(class_name)
            # 遍历该类别下的所有文件
            for file_name in os.listdir(class_path):
                if not file_name.endswith('.npy'):
                    continue
                file_path = os.path.join(class_path, file_name)
                # 加载npy文件
                data = np.load(file_path)
                # 检查数据形状是否符合预期
                if len(data.shape) != 2:  # 假设期望形状是(通道, 特征, 时间步)
                    raise ValueError(f"数据形状不符合预期: {data.shape}")
                # 将数据分割成512长度的窗口
                num_windows = data.shape[1] // 512
                for i in range(num_windows):
                    start = i * 512
                    end = (i + 1) * 512
                    window_data = data[:, start:end]
                    # 确保窗口大小正好是512
                    if window_data.shape[1] != 512:
                        continue
                    self.data.append(window_data)
                    self.labels.append(label)
        # 转换为numpy数组便于后续处理
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        # print(f"总样本数: {len(self.data)}")
        # print(f"类别0样本数: {np.sum(self.labels == 0)}")
        # print(f"类别1样本数: {np.sum(self.labels == 1)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取对应索引的数据和标签
        data = self.data[index]
        label = self.labels[index]
        # 转换为torch张量
        data_tensor = torch.FloatTensor(data)
        label_tensor = torch.LongTensor([label])  # 使用LongTensor分类任务
        return data_tensor, label_tensor

dataloader_with_label = torch.utils.data.DataLoader(dataset=Down_train_data(), batch_size=8, shuffle=True)

# for batch_x, batch_y in dataloader:
#     print(len(dataloader))
#     print(batch_x.shape)  # 应该是(batch_size, 通道数=22, 512)
#     print(batch_y.shape)  # 应该是(batch_size,)
#     break
# 总样本数: 530
# 类别0样本数: 263
# 类别1样本数: 267