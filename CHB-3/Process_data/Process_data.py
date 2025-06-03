import os
import mne                                                                     # 读edf文件
import numpy as np
from scipy.stats import iqr                                                    # 数据标准化
import warnings
"""
            文件注释：
                    1、通道23个选择22个，最后一个抛弃（原因：重复）
                    2、单位转换（V → μV） 
                    3、滤波0.5-45 HZ
                    4、Z-score标准化

"""
def process_data(file_path):
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