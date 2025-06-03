import torch.nn.functional as F
import torch.nn as nn
import torch
from Model.block_split import block_split
"""
            第一阶段：（块内预测，局部特征提取）
                    token:22*16=352,一个时间段里面有 8个token
                    块内嵌入模块（带掩码预测的通道独立LSTM）
                    输入: (batch_size, channel, block, seq_len) = (8, 22, 8, 64)
                    输出: (batch_size, channel, block, hidden_dim) = (8, 22, 8, 16)       
            第二阶段:（块间预测，全局特征提取）
                    输入: (batch_size, block, channel*seq_len) = (8, 8, 352)
                    输出: (batch_size, cls, feature) = (8,1,352)
            -------------------------------------------------------------------------
            操作:
                    (8, 22, 8, 16)->(8, 8, 352) : 
                                tokens = token.permute(0, 2, 1, 3)
                                tokens = tokens.reshape(8, 8, 22 * 16)
"""


class block_Embedding(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=4, num_channels=22, mask_ratio=0.3):
        super(block_Embedding, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.mask_radio = mask_ratio
        # 每个通道的独立模块
        self.channel_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True
            )
            for _ in range(num_channels)
        ])
        # 共享预测头
        self.predictor = nn.Linear(hidden_dim, input_dim)
        # 参数初始化
        self._init_weights()
    def _init_weights(self):
        """初始化LSTM和预测头的权重"""
        for lstm in self.channel_lstms:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        nn.init.xavier_normal_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, x):
        batch_size, num_channels, num_blocks, seq_len = x.shape
        # 添加特征维度 [batch, channels, blocks, seq_len, 1]
        x = x.unsqueeze(-1)
        all_outputs = []
        total_loss = 0.0
        for ch in range(num_channels):
            # 获取当前通道数据 [batch, blocks, seq_len, 1]
            channel_data = x[:, ch, :, :, :]
            # 合并batch和blocks维度 [batch*blocks, seq_len, 1]
            channel_data = channel_data.reshape(-1, seq_len, 1)
            # 生成随机掩码
            with torch.no_grad():
                mask = torch.rand(channel_data.shape[:-1], device=x.device) < self.mask_radio
                masked_input = channel_data.clone()
                masked_input[mask] = 0.0  # 零值掩码
                original_values = channel_data[mask]
            # LSTM处理 [batch*blocks, seq_len, hidden_dim]
            lstm_out, _ = self.channel_lstms[ch](masked_input)
            # 掩码预测任务
            if mask.any():
                pred_values = self.predictor(lstm_out[mask])
                channel_loss = F.mse_loss(pred_values, original_values)
                total_loss += channel_loss
            # 恢复维度 [batch, blocks, seq_len, hidden_dim]
            lstm_out = lstm_out.reshape(batch_size, num_blocks, seq_len, -1)
            all_outputs.append(lstm_out)
        # 合并所有通道 [batch, channels, blocks, seq_len, hidden_dim]
        output = torch.stack(all_outputs, dim=1)
        # 取最后16个时间步并平均 [batch, channels, blocks, hidden_dim]
        output = output[:, :, :, -16:, :].mean(dim=3)
        # 计算平均损失
        total_loss /= num_channels
        return output, total_loss


# --------------------------------------------------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=352, hidden_dim=512, num_layers=4, mask_ratio=0.3):
        super().__init__()
        # 参数配置
        self.mask_ratio = mask_ratio  # 掩码比例
        self.num_blocks = 8  # 固定8个块（可根据需求调整）
        # 1. 投影层
        self.proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        # 2. 位置编码（可学习）
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_blocks + 1, hidden_dim))
        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 4. 特殊Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 分类Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 掩码Token
        # 5. 重建头
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, return_loss=True):
        # 输入: x: (batch_size=8, num_blocks=8, input_dim=352)
        batch_size = x.size(0)
        x_original = x.clone()  # 保存原始输入用于损失计算
        # --- 1. 准备输入序列 ---
        # 线性投影
        x = self.proj(x)  # (8, 8, hidden_dim)
        # 添加CLS Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (8, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (8, 9, hidden_dim)
        # --- 2. 掩码操作 ---
        if return_loss:
            # 生成随机掩码（不掩码CLS Token）
            num_masked = int(self.mask_ratio * self.num_blocks)
            rand_indices = torch.rand(batch_size, self.num_blocks).argsort(dim=1)
            mask_indices = rand_indices[:, :num_masked] + 1  # 偏移1以跳过CLS Token
            unmask_indices = rand_indices[:, num_masked:] + 1
            # 创建掩码版本而不修改原始x
            x_masked = x.clone()
            mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)
            batch_indices = torch.arange(batch_size).unsqueeze(-1)
            x_masked[batch_indices, mask_indices, :] = mask_tokens
            # 记录掩码位置（用于损失计算）
            mask_flag = torch.zeros(batch_size, self.num_blocks + 1, 1, dtype=torch.bool, device=x.device)
            mask_flag[batch_indices, mask_indices] = True
        else:
            x_masked = x
            mask_flag = None
        # --- 3. 添加位置编码 ---
        x_masked = x_masked + self.pos_embedding[:, :x_masked.size(1), :]
        # --- 4. Transformer编码 ---
        encoded = self.transformer(x_masked)  # (8, 9, hidden_dim)
        # --- 5. 输出处理 ---
        # 提取CLS Token（用于下游任务）
        cls_output = encoded[:, 0, :].unsqueeze(1)  # (8, 1, hidden_dim)
        cls_output = self.reconstruction_head(cls_output)  # (8, 1, 352)
        # --- 6. 损失计算 ---
        loss = None
        if return_loss and mask_flag is not None:
            # 只计算被掩码块的重建损失
            pred = self.reconstruction_head(encoded[:, 1:, :])  # (8, 8, 352)
            target = x_original[:, :, :]  # 使用原始输入
            # 仅对掩码位置计算MSE
            mask_flag = mask_flag[:, 1:, :]  # 跳过CLS Token
            loss = F.mse_loss(
                pred[mask_flag.expand_as(pred)],
                target[mask_flag.expand_as(target)],
                reduction='mean'
            )
        return cls_output, loss

# --------------------------------------------------------------------------------------------------------------------

class combing_model(nn.Module):
    def __init__(self, mask_ratio_1=0.3, mask_ratio_2=0.3, loss_rate=0.5):
        super().__init__()
        self.mask1 = mask_ratio_1
        self.mask2 = mask_ratio_2
        self.loss_rate = loss_rate
        self.block_Embedding = block_Embedding(mask_ratio=self.mask1)
        self.Transformer = TransformerEncoder(mask_ratio=self.mask2)

    def forward(self, x):
        x = block_split(x)
        token, loss1 = self.block_Embedding(x)
        # 维度变换
        token = token.permute(0, 2, 1, 3)
        token = token.reshape(token.shape[0], 8, 22 * 16)
        cls, loss2 = self.Transformer(token)
        all_loss = loss1*self.loss_rate + loss2*(1-self.loss_rate)
        # 维度变换
        cls = cls.squeeze(dim=1)  # 输出形状: (8, 352)
        return cls, all_loss



"""
    测试代码：
            model = combing_model()
            x = torch.randn(8, 22, 8*64)  # 输入数据
            output, loss = model(x)
            print(output.shape)
            print(loss.item())   
---------------------------------------------------------
    x    = torch.Size([8, 22, 512])
    cls  = torch.Size([8, 352])
    loss = 0.6432350873947144
"""
