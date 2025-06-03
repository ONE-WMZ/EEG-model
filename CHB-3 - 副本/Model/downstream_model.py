import torch.nn as nn
import torch

# 复杂分类模型
class Downstream_Class(nn.Module):
    def __init__(self, input_dim=352):
        super(Downstream_Class, self).__init__()
        # 特征处理分支
        self.class_x = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.class_x(x)
        return logits.squeeze()

""" 测试代码：
        model = Downstream_Class()
        x = torch.randn(8, 1, 352)  # 输入数据
        y = model(x)
        print(y)
"""

