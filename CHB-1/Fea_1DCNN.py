import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * (512 // 4), 32)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义解码器模型
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 全连接层：输入维度=32，输出维度=128 * (512 // 4)
        self.fc = nn.Linear(32, 128 * (512 // 4))
        # 反卷积层1：输入通道数=128，输出通道数=64，卷积核大小=3
        self.deconv1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        # 反卷积层2：输入通道数=64，输出通道数=1，卷积核大小=3
        self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)

    def forward(self, x):
        # 全连接层
        x = self.fc(x)
        x = torch.relu(x)
        # 调整形状以适应反卷积层
        x = x.view(x.size(0), 128, (512 // 4))
        # 反卷积层+ReLU激活函数
        x = torch.relu(self.deconv1(x))
        # 最终反卷积层
        x = self.deconv2(x)
        return x

# 自动编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        fea = self.encoder(x)
        x = self.decoder(fea)
        return x, fea
