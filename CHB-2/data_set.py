import numpy as np
import torch
from torch.utils.data import Dataset
from data_tool import after_edf as get_data       # ***
from data_tool import root_path as root_path
from data_tool import file_list as file_list

"""
        数据说明:
            BatchSize = 8
            Seq_len = 512 (=256*2) (2s)
            channel_num =  22
            数据集分为 10个 （ 8个训练 ：2个测试 ）
            get_item.shape = (8,22,512) 
            每个被试-随机采样不同任务 --- 模拟多任务
            
"""
class train_dataset(Dataset):
    def __init__(self, file_path_list):   # 8*22*1h
        self.file_path = file_path_list
        self.data = []
        for file in self.file_path:
            data = get_data(file)          # after_edf
            if data.shape[1] > 3600*256:
                data = data[:, data.shape[1]-(3600*256):data.shape[1]]    # 数据文件太长时进行裁剪
            self.data.append(data)
        self.data = np.array(self.data)

    def __len__(self):
        return 1800

    def __getitem__(self, item):
        window_size = 512
        start_pos = item * window_size
        end_pos = (item+1) * window_size
        data_x = self.data[:, :, start_pos:end_pos]
        data_x = torch.FloatTensor(data_x)
        return data_x


def train_file_path():
    train_list = []
    for file_list_x in file_list[0:8]:
        file_list_id = root_path + file_list_x
        train_list.append(file_list_id)
    return train_list

train_set = train_dataset(train_file_path())


