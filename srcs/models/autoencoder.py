import torch.nn as nn


def conv_block_2d(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class AutoEncoder2D(nn.Module):
    def __init__(self, in_channels=6, base=16, out_channels=2):
        super().__init__()

        self.enc1 = conv_block_2d(in_channels, base)
        self.down1 = nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = conv_block_2d(base * 2, base * 2)

        self.down2 = nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1)
        self.enc3 = conv_block_2d(base * 4, base * 4)

        self.down3 = nn.Conv2d(base * 4, base * 8, kernel_size=3, stride=2, padding=1)
        self.enc4 = conv_block_2d(base * 8, base * 8)

        self.down4 = nn.Conv2d(base * 8, base * 16, kernel_size=3, stride=2, padding=1)
        self.bottleneck = conv_block_2d(base * 16, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block_2d(base * 8, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block_2d(base * 4, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block_2d(base * 2, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = conv_block_2d(base, base)

        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.enc4(self.down3(x3))
        xb = self.bottleneck(self.down4(x4))

        y = self.dec4(self.up4(xb))
        y = self.dec3(self.up3(y))
        y = self.dec2(self.up2(y))
        y = self.dec1(self.up1(y))
        y = self.out(y)
        return y
