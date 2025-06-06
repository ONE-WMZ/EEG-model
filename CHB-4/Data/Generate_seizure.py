import pandas as pd
import numpy as np
import os
from Process_data.Preprocess import process_data    # 导入数据处理函数

"""
        划分数据：（保存为csv格式）
            训练集：
                label_0:正常(每个人采集20分钟)
                label_1:异常(每个人的癫痫数据放在一块)
            验证集：
                label_0:正常(每个人采集20分钟)【未训练过的】
                label_1:异常(每个人的癫痫数据放在一块)【未训练过的】
"""
# 数据集根目录
root_path = "F:/CHB_MIT/chb-mit-scalp-eeg-database-1.0.0/"

seizure_info = '../Process_data/seizure_info.csv'   # 文件地址
seizure_all = pd.read_csv(seizure_info)        # 读文件

all_row = len(seizure_all)                     # 获取行数
# 划分训练集 & 测试集
train_row = 159                                     # 20个被试
test_row = all_row - train_row                      # 4个被试


# 保存路径设置
train_dir = 'train'
label_0_dir = 'label_0'
label_1_dir = 'label_1'


def generate_data(user_id):
    all_seizure_data = []
    user_list = seizure_all[seizure_all['File Name'].str.startswith(user_id)]
    print(user_list)
    for i in range(len(user_list)):
        file_name = root_path + str(user_id) + '/' + user_list.iloc[i]['File Name']
        start = user_list.iloc[i]['Seizure Start Time (seconds)']
        end = user_list.iloc[i]['Seizure End Time (seconds)']
        # 处理数据
        data = process_data(file_name)
        if data is None:
            continue
        else:
            # 分割数据
            seizure_data = data[:, start*200:end*200]
            all_seizure_data.append(seizure_data)

    # 合并数据 构建保存路径
    merged_data = np.concatenate(all_seizure_data, axis=1)
    # 保存数据
    save_path = os.path.join(train_dir, label_1_dir, f'{user_id}.npy')
    np.save(save_path, merged_data)
    print('✅', user_id, ':处理完成.\t', merged_data.shape)

def train_gen():
    print('开始处理训练数据...')
    for i in range(12, 25):
        user_id = f'chb{i:02d}'
        generate_data(user_id)

# train_gen()