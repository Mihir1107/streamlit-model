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


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avgpool(x)
        y = self.fc(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        hidden = max(8, in_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.conv(x)
        return x * att


class RCAB(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
        )
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.res_scale = 0.1

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res * self.res_scale


class ResidualGroup(nn.Module):
    def __init__(self, channels, n_rcab=4):
        super().__init__()
        layers = [RCAB(channels) for _ in range(n_rcab)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x) + x


class LearnedUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(in_channels, out_channels * (scale * scale), kernel_size=3, padding=1)
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, target_size=None):
        x = self.proj(x)
        x = F.pixel_shuffle(x, self.scale)
        x = self.post(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class DualEDSR(nn.Module):
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
            nn.ReLU(inplace=True)
        )
        self.convOut = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, xT, xO):
        fT = F.relu(self.convT(xT))
        fO = F.relu(self.convO(xO))
        fT = self.resBlocksT(fT)
        fO = self.resBlocksO(fO)
        fT_up = F.interpolate(fT, size=(fO.shape[2], fO.shape[3]),
                              mode='bilinear', align_corners=False)
        f = torch.cat([fT_up, fO], dim=1)
        f = F.relu(self.convFuse(f))
        f = self.refine(f)
        out = self.convOut(f)
        return out


class DualEDSRPlus(nn.Module):
    def __init__(self, n_resgroups=4, n_rcab=4, n_feats=64, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.n_feats = n_feats

        self.convT_in = nn.Conv2d(1, n_feats, 3, padding=1)
        self.convO_in = nn.Conv2d(3, n_feats, 3, padding=1)

        self.t_groups = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])
        self.o_groups = nn.Sequential(*[ResidualGroup(n_feats, n_rcab) for _ in range(n_resgroups)])

        self.t_upsampler = LearnedUpsampler(n_feats, n_feats, scale=upscale)

        self.convFuse = nn.Conv2d(2 * n_feats, n_feats, kernel_size=1)
        self.fuse_ca = ChannelAttention(n_feats)
        self.fuse_sa = SpatialAttention(n_feats)

        self.refine = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convOut = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xT, xO):
        fT = F.relu(self.convT_in(xT))
        fO = F.relu(self.convO_in(xO))

        fT = self.t_groups(fT)
        fO = self.o_groups(fO)

        fT_up_raw = self.t_upsampler(fT)
        target_hw = (fO.shape[2], fO.shape[3])
        fT_up = F.interpolate(fT_up_raw, size=target_hw, mode="bilinear", align_corners=False)

        f = torch.cat([fT_up, fO], dim=1)
        f = F.relu(self.convFuse(f))
        f = self.fuse_ca(f)
        f = self.fuse_sa(f)
        f = self.refine(f)
        out = self.convOut(f)
        return out
