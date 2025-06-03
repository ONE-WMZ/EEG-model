import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out += residual
        return F.relu(out)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Downsample path
        self.down1 = nn.Sequential(
            ResidualBlock(22, 64),
            nn.MaxPool1d(2)
        )
        self.down2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool1d(2)
        )
        self.down3 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool1d(2)
        )
        self.down4 = nn.Sequential(
            ResidualBlock(256, 512),
            nn.MaxPool1d(2)
        )
        # Bottleneck
        self.bottleneck = ResidualBlock(512, 512)
        # Final projection to (B, 1, 256)
        self.final_proj = nn.Conv1d(512, 256, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, 22, 64)
        x = self.down1(x)  # (B, 64, 32)
        x = self.down2(x)  # (B, 128, 16)
        x = self.down3(x)  # (B, 256, 8)
        x = self.down4(x)  # (B, 512, 4)
        x = self.bottleneck(x)  # (B, 512, 4)
        # Global average pooling and projection
        x = torch.mean(x, dim=2, keepdim=True)  # (B, 512, 1)
        x = self.final_proj(x)  # (B, 256, 1)
        return x.permute(0, 2, 1)  # (B, 1, 256)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Initial expansion from (B, 1, 256) to (B, 512, 4)
        self.init_expand = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='nearest')
        )
        # Upsample path
        self.up1 = nn.Sequential(
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.up2 = nn.Sequential(
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.up3 = nn.Sequential(
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.up4 = nn.Sequential(
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        # Final projection to (B, 22, 64)
        self.final_proj = nn.Conv1d(64, 22, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, 1, 256)
        x = x.permute(0, 2, 1)  # (B, 256, 1)
        x = self.init_expand(x)  # (B, 512, 4)
        x = self.up1(x)  # (B, 512, 8)
        x = self.up2(x)  # (B, 256, 16)
        x = self.up3(x)  # (B, 128, 32)
        x = self.up4(x)  # (B, 64, 64)
        x = self.final_proj(x)  # (B, 22, 64)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        loss = F.mse_loss(x, decoded)
        return decoded, loss