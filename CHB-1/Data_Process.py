import os
import mne                                                                     # 读edf文件
import numpy as np
from scipy.stats import iqr                                                    # 数据标准化
import warnings                                                                # 忽略警告


"""  重要须知：(1)
                            1、数据预处理：
                                    1-1、读文件（edf）
                                    1-2、滤波（1-40）
                                    1-3、标准化（Robust 标准化）
"""


# ---------------------------------------------------------------------------------------------------------

def file_list(path):                                                             # 读文件列表
    filelist = os.listdir(path)
    return filelist

# ---------------------------------------------------------------------------------------------------------

def get_address():                                                               # 取地址
    all_edf_address = []                                                                # 所有的edf文件(686个)
    all_seizures_address = []                                                           # 癫痫数据集合（141个）
    path = 'F:/CHB_MIT/chb-mit-scalp-eeg-database-1.0.0'                                # 数据集地址
    file_user_raw = file_list(path)                                                     # len=30
    file_user = file_user_raw[1:25]                                                     # 24个
    for file_user_case in file_user:
        file_user_case_raw = file_list(path+'/'+file_user_case)
        for file_user_case_ID in file_user_case_raw:
            if file_user_case_ID[-3:] == 'edf':
                all_edf_address.append(path+'/'+file_user_case+'/'+file_user_case_ID)
            elif file_user_case_ID[-8:] == 'seizures':
                all_seizures_address.append(path+'/'+file_user_case+'/'+file_user_case_ID)

    return all_edf_address, all_seizures_address

data_address, seizure_address = get_address()

# ---------------------------------------------------------------------------------------------------------

def robust_normalization(data):                                                    # Robust 标准化
    median = np.median(data, axis=0)                                                            # 计算中位数
    iqr_val = iqr(data, axis=0)                                                                 # 计算四分位距
    normalized_data = (data - median) / iqr_val
    return normalized_data

# ---------------------------------------------------------------------------------------------------------

def data_preprocess(file_path):                                                    # 数据预处理
    warnings.filterwarnings("ignore", category=RuntimeWarning)                         # 忽略警告
    raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)                   # 读文件
    picks = mne.pick_types(raw_data.info, eeg=True)                                          # 选择通道
    filtered_data = raw_data.filter(1, 40, picks=picks, verbose=False)           # 滤波(1-40)
    data, time = filtered_data.get_data(return_times=True)                                   # 加载数据和时间
    data = data * 1000000
    normalized_data = robust_normalization(data.T).T                                         # 标准化-data
    return normalized_data

# ---------------------------------------------------------------------------------------------------------
# 测试代码：
# data_ = data_preprocess(data_address[0])
# print(type(data_))
# print(data_.shape)

