from Process_data.Preprocess import process_data
import numpy as np
import os

# 数据集根目录
root_path = "F:/CHB_MIT/chb-mit-scalp-eeg-database-1.0.0/"
# 每个人选20分钟的正常文件
normal_list = ['chb01_01.edf', 'chb02_01.edf', 'chb03_05.edf', 'chb04_01.edf', 'chb05_01.edf', 'chb06_02.edf',
               'chb07_01.edf', 'chb08_03.edf', 'chb09_01.edf', 'chb10_01.edf', 'chb11_01.edf', 'chb12_19.edf',
               'chb13_02.edf', 'chb14_42.edf', 'chb15_02.edf', 'chb16_01.edf', 'chb17a_05.edf', 'chb18_01.edf',
               'chb19_01.edf', 'chb20_01.edf', 'chb21_01.edf', 'chb22_01.edf', 'chb23_10.edf']


# 保存路径设置
train_dir = 'train'
label_0_dir = 'label_0'


def generate_normal_data():
    for i in range(18, 20):
        file_path = root_path+f'chb{i:02d}/'+normal_list[i-1]
        # print(file_path)
        data = process_data(file_path)
        if data is None:
            continue
        else:
            # 分割数据
            seizure_data = data[:, 0:20*60*200]
            # 保存数据
            save_path = os.path.join(train_dir, label_0_dir, f'chb{i:02d}.npy')
            np.save(save_path, seizure_data)
            print('✅', i, ':处理完成.\t', seizure_data.shape)


# generate_normal_data()

# %% 24单独处理
# normal_24 = 'chb24_13.edf'
# file_path = root_path+f'chb24/'+normal_24
# print(file_path)
# data = process_data(file_path)
# # 分割数据
# seizure_data = data[:, 0:20*60*200]
# # 保存数据
# save_path = os.path.join(train_dir, label_0_dir, f'chb24.npy')
# np.save(save_path, seizure_data)
# print('✅', ':处理完成.\t', seizure_data.shape)