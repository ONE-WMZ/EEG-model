import mne
import numpy as np
import warnings

"""
    数据预处理:
        1. 读取原始EDF文件
        2. 重采样到200Hz
        3. 删除重复的通道（根据名称）[可选择指定]
        4. 将数据从 V 转换为 μV，并更新单位信息
        5. 带通滤波：0.5 - 40 Hz
        6. 获取数据并进行 Z-score 标准化（逐通道）
        得到的数据形状：shape = (n_channels, n_times) = (6, 720000)
"""


def process_data(file_path):
    # 忽略所有 MNE 的 info 日志
    mne.set_log_level('WARNING')
    # 忽略运行时警告
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. 读取原始EDF文件
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    # 2. 重采样到200Hz
    raw.resample(200.0)

    # 3. 删除重复的通道（根据名称）
    # ch_names = raw.info['ch_names']
    # unique_names, indices = np.unique(ch_names, return_index=True)
    # channels_to_drop = [ch for idx, ch in enumerate(ch_names) if idx not in indices]
    # if channels_to_drop:
    #     raw.drop_channels(channels_to_drop)
    #     print(f"Dropped duplicate channels: {channels_to_drop}")
    # else:
    #     print("No duplicate channels found.")

    # 4. 将数据从 V 转换为 μV，并更新单位信息
    raw.apply_function(lambda x: x * 1e6)  # V → μV

    # 更新通道单位信息
    for ch in raw.info['chs']:
        ch['unit'] = 105  # 对于μV，MNE使用105作为单位代码

    # 5. 带通滤波：0.5 - 40 Hz
    raw.filter(l_freq=0.5, h_freq=40, method='fir', fir_window='hamming', n_jobs=2)

    # 6. 挑选指定的通道（六个）
    selected_channels = ['F7-T7', 'T7-P7', 'F8-T8', 'T8-P8', 'FZ-CZ', 'CZ-PZ']  # 'T8-P8-0'
    # 获取当前数据中存在的通道名（注意大小写和命名一致性）
    available_channels = raw.info['ch_names']
    # 检查所有 selected_channels 是否都存在于当前数据中
    missing_channels = [ch for ch in selected_channels if ch not in available_channels]

    if missing_channels:
        print(f"文件缺少以下通道：{missing_channels}，跳过此文件。")
        return None
    else:
        # 所有通道都存在，复制并挑选通道
        raw_copy = raw.copy().pick(selected_channels)


    # 获取数据并进行 Z-score 标准化（逐通道）
    data = raw_copy.get_data()  # shape: (n_selected_channels, n_times)
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    data_normalized = (data - data_mean) / (data_std + 1e-8)  # 防止除零

    # 返回处理后数据
    # 返回所选通道的数据
    return data_normalized  # shape: (6, n_times)



""""
-------------------------------------------------
        导联	    覆盖区域	    癫痫类型
-------------------------------------------------
        F7-T7	左前颞叶	    局灶性（颞叶）
        T7-P7	左中后颞叶	颞叶扩散性放电
        F8-T8	右前颞叶	    局灶性（颞叶）
        T8-P8	右中后颞叶	颞叶扩散性放电
        FZ-CZ	额-中央中线	全面性/额叶中线发作
        CZ-PZ	中央-顶中线	全面性发作
-------------------------------------------------
        【覆盖 >90% 的常见癫痫发作类型（颞叶+全面性）】
-------------------------------------------------        
所有通道：【FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3,
        P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8-0,
        P8-O2, FZ-CZ,CZ-PZ, P7-T7, T7-FT9, FT9-FT10, FT10-T8】22个
"""

"""
    1、硬件的蓝牙问题（数据量越大，时间耗费越大）
    2、因为一导致的模型数据问题（因为时间成本的问题，将通道进行压缩）
    3、前端，服务器，设备之间的实时交互问题
"""