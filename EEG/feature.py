import pandas as pd
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt


#  (62,42,5)---[电极，时间片段，个数]
#  ------------------------------------------------------------------
#  关键字（eeg_feature_smooth）
#  -------------------
#  'de_movingAve1', 'de_LDS1', 'psd_movingAve1', 'psd_LDS1',
#   ......
#  'de_movingAve24', 'de_LDS24', 'psd_movingAve24', 'psd_LDS24'
#  --------------------------------------------------------------------
#  || DE:（差分熵）|| PSD:（功率谱密度）||
#  || movingAve:（移动平均）|| LDS:（线性动态系统）||
#  || 数字：（24段视频）||
#  --------------------------------------------------------------------
