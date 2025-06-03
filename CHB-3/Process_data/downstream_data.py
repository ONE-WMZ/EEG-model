import numpy as np
from Process_data import process_data

"""
    文件数据：（测试数据）：
            裁剪数据 ——> 确保数据平衡
            -----------------------------------------
            | 被试间：【1、2、3、4、5、6、7、8】        |
            -----------------------------------------       
            | 跨被试：【9、10】                       |
            -----------------------------------------
            
"""

root_path = 'F:/CHB_MIT/chb-mit-scalp-eeg-database-1.0.0/'
file_list = ['chb01/chb01_03.edf',  # 1h
             'chb02/chb02_19.edf',  # 1h
             'chb03/chb03_01.edf',  # 1h
             'chb04/chb04_05.edf',  # 2h39m
             'chb05/chb05_06.edf',  # 1h
             'chb06/chb06_09.edf',  # 4h
             'chb07/chb07_12.edf',  # 4h
             'chb08/chb08_02.edf',  # 1h
        # -----------------------------------------
             'chb09/chb09_06.edf',  # 4h
             'chb10/chb10_12.edf']  # 2h
# 单位：秒（s）
seizure_sta = [2996, 3369, 362, 7804, 417, 12500, 4920, 2670, 12231, 6313]
seizure_end = [3036, 3378, 414, 7853, 532, 12516, 5006, 2841, 12295, 6348]

for i in range(8):
    data_path = root_path+file_list[i]
    data = process_data(data_path)
    # 裁剪数据(标签数据)
    star = seizure_sta[i]
    end  = seizure_end[i]
    label_1 = data[:, star*256:end*256]   # 标签=1
    print(label_1.shape)
    label_0 = data[:, star*256-label_1.shape[1]:star*256-1]
    # np.save('data/test_data/1/'+'label_1-'+str(i+1)+'.npy', label_1)
    # np.save('data/test_data/0/'+'label_0-'+str(i+1)+'.npy', label_0)
