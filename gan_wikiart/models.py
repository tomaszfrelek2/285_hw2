"""DCGAN generator and discriminator for 64x64 images (Radford et al., 2016)."""
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        def up(i, o):
            return [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),  # 4x4
            *up(ngf * 8, ngf * 4), *up(ngf * 4, ngf * 2), *up(ngf * 2, ngf),                               # 32x32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh())                                    # 64x64

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        def down(i, o):
            return [nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),  # 32x32
            *down(ndf, ndf * 2), *down(ndf * 2, ndf * 4), *down(ndf * 4, ndf * 8),  # 4x4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))  # logit (sigmoid is inside the loss)

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
