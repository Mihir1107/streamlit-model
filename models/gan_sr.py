import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * 0.1


class GANGenerator(nn.Module):
    """
    Standalone generator for thermal SR.
    Inputs:
      xT: [B,1,H,W] low-res thermal
      xO: [B,3,H,W] high-res optical
    Output:
      [B,1,H,W] super-res thermal
    """

    def __init__(self, n_resblocks=8, n_feats=64, upscale=3):
        super().__init__()
        self.upscale = upscale
        self.convT = nn.Conv2d(1, n_feats, 3, padding=1)
        self.convO = nn.Conv2d(3, n_feats, 3, padding=1)
        self.resBlocksT = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.resBlocksO = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)])
        self.convFuse = nn.Conv2d(2 * n_feats, n_feats, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convOut = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, xT, xO):
        fT = F.relu(self.convT(xT))
        fO = F.relu(self.convO(xO))
        fT = self.resBlocksT(fT)
        fO = self.resBlocksO(fO)
        fT_up = F.interpolate(fT, size=(fO.shape[2], fO.shape[3]), mode="bilinear", align_corners=False)
        f = torch.cat([fT_up, fO], dim=1)
        f = F.relu(self.convFuse(f))
        f = self.refine(f)
        out = self.convOut(f)
        return out


class PatchDiscriminator(nn.Module):
    """Small PatchGAN discriminator on concatenated thermal+optical inputs."""

    def __init__(self, in_ch=4):
        super().__init__()

        def block(c_in, c_out, stride=2, norm=True):
            layers = [
                nn.Conv2d(c_in, c_out, 4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if norm:
                layers.insert(1, nn.BatchNorm2d(c_out))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_ch, 64, stride=2, norm=False),
            block(64, 128, stride=2, norm=True),
            block(128, 256, stride=2, norm=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1),  # logits map
        )

    def forward(self, x):
        return self.net(x)


def build_gan(generator=None):
    gen = generator if generator is not None else GANGenerator()
    disc = PatchDiscriminator()
    return gen, disc
