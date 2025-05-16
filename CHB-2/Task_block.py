import torch
import torch.nn as nn
import torch.nn.functional as F



# ————————————————————————————————————————————    任务-1（Task-1）    ———————————————————————————————————————————————————
"""
        任务一 ：对比（添加噪声）  --> 交叉熵损失
        数据维度 (batch_size,feature_dim) = (8,64)
"""
def task_1(x, mean=0., std=0.1):   # 添加高斯噪声   【loss 最低 = 0.66】
    noise = torch.randn_like(x) * std + mean
    return x + noise

def contrastive_loss(x1, x2, temperature=0.5):
    batch_size = x1.shape[0]
    # 归一化特征
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)
    # 相似度矩阵 [B, B]
    sim = torch.mm(x1, x2.t()) / temperature
    # 正样本：对角线位置
    pos = torch.diag(sim).unsqueeze(1)  # [B, 1]
    # 负样本：其他位置
    neg = sim[~torch.eye(batch_size, dtype=bool)].view(batch_size, -1)  # [B, B-1]
    # 合并 logits [B, B]
    logits = torch.cat([pos, neg], dim=1)
    # 标签：正样本在位置 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=x1.device)
    return F.cross_entropy(logits, labels)


# ———————————————————————————————————————————    任务-2（Task-2）   —————————————————————————————————————————————————————
def task_2(x, ratio=0.5):
    return x
# ————————————————————————————————————————————   任务-3（Task-3）  ——————————————————————————————————————————————————————

# ————————————————————————————————————————————      辅助函数       ——————————————————————————————————————————————————————

