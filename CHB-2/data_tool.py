import os
import mne                                                                     # 读edf文件
import numpy as np
from scipy.stats import iqr                                                    # 数据标准化
import warnings

"""
            文件注释：
                    1、每个人采用一个数据文件进行训练（1h）选择带有癫痫发作的数据文件
                    2、通道23个选择22个，最后一个抛弃（原因：重复）
                    3、单位转换（V → μV）  ***
                    3、滤波0.5-45 HZ
                    4、Z-score标准化
                    （用于文件data_set）
"""


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

root_path = 'F:/CHB_MIT/chb-mit-scalp-eeg-database-1.0.0/'
file_list = ['chb01/chb01_03.edf',
             'chb02/chb02_19.edf',
             'chb03/chb03_01.edf',
             'chb04/chb04_05.edf',
             'chb05/chb05_06.edf',
             'chb06/chb06_01.edf',
             'chb07/chb07_12.edf',
             'chb08/chb08_02.edf',
             'chb09/chb09_06.edf',
             'chb10/chb10_12.edf']

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def after_edf(file_path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略警告
    raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)  # 读文件
    picks = mne.pick_types(raw_data.info, eeg=True, exclude=[raw_data.ch_names[-1]])  # 抛弃最后一个
    raw_selected = raw_data.copy().pick(picks)  # 选择通道

    for idx in picks:                                     # 统一乘以1e6 (V → μV) 并更新单位编码
        raw_selected._data[idx] *= 1e6                    # 修改数据
        raw_selected.info['chs'][idx]['unit'] = 101       # 107=V → 101=μV
        raw_selected.info['chs'][idx]['cal'] = 1.0        # 重置校准因子

    mne.set_log_level(verbose='WARNING')  # 只显示 WARNING 和 ERROR，忽略 INFO
    raw_filtered = raw_selected.filter(l_freq=0.5, h_freq=40, method='fir', fir_window='hamming', n_jobs=2)  # 滤波
    get_data = raw_filtered.get_data()
    data_normalized = (get_data - np.mean(get_data)) / np.std(get_data)  # Z-score标准化: 均值0，方差1
    return data_normalized

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def before(file_path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略警告
    raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)  # 读文件
    picks = mne.pick_types(raw_data.info, eeg=True, exclude=[raw_data.ch_names[-1]])  # 抛弃最后一个
    raw_selected_5 = raw_data.copy().pick(picks)  # 选择通道
    get_data = raw_selected_5.get_data()
    mne.set_log_level(verbose='WARNING')  # 只显示 WARNING 和 ERROR，忽略 INFO
    data_normalized = (get_data - np.mean(get_data)) / np.std(get_data)  # Z-score标准化: 均值0，方差1
    return data_normalized



# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
""" --- 测试代码 ---
    path = root_path+file_list[0]
    data = after_edf(path)
    
"""

