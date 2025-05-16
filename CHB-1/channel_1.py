import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(EncoderDecoder, self).__init__()

        # 定义编码器的双向LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

        # 解码器的隐藏状态维度是双向LSTM输出的两倍
        decoder_input_size = 2 * hidden_size

        # 定义解码器的单向LSTM
        self.decoder = nn.LSTM(decoder_input_size, hidden_size*2, num_layers=num_layers, batch_first=True)

        # 输出层，将隐藏状态映射回原始输入尺寸
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        # 编码过程
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # 将双向LSTM的两个方向的隐藏状态拼接起来作为解码器的初始隐藏状态
        hidden = torch.cat((hidden[:1, :, :], hidden[1:, :, :]), dim=2)
        cell = torch.cat((cell[:1, :, :], cell[1:, :, :]), dim=2)

        # 解码过程
        decoder_outputs, _ = self.decoder(encoder_outputs, (hidden, cell))

        # 输出层
        outputs = self.fc(decoder_outputs)

        return outputs, hidden
