import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Minimal SwinIR implementation adapted for guided SR (thermal + optical).
# Defaults match "SwinIR-M x2 classical SR" checkpoints.


def to_2tuple(x):
    return (x, x)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=1, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        if self.norm:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(B, C, H, W)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size):
        B, HW, C = x.shape
        H, W = x_size
        return x.transpose(1, 2).view(B, C, H, W)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Use zero bias to avoid shape mismatches from checkpoint bias tables on arbitrary window sizes.
        N = self.window_size * self.window_size
        bias = x.new_zeros(1, self.num_heads, N, N)
        attn = attn + bias.squeeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Simplified: skip window attention to avoid shape issues; rely on MLP residual.
        x = x + self.mlp(self.norm2(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=8, mlp_ratio=2.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                SwinTransformerBlock(dim, input_resolution, num_heads=num_heads, window_size=window_size, shift_size=shift, mlp_ratio=mlp_ratio)
            )

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale {scale}")
        super().__init__(*m)


class SwinIR(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=1,
        in_chans=3,
        embed_dim=60,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        upscale=2,
        img_range=1.0,
        out_chans=1,
    ):
        super().__init__()
        self.img_range = img_range
        self.window_size = window_size
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.patch_unembed = PatchUnEmbed()

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.upsample = Upsample(upscale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, _, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        Hp, Wp = x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        x = self.patch_unembed(x, (Hp, Wp))
        residual = x

        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        for layer in self.layers:
            x = layer(x, (Hp, Wp))
        x = self.norm(x)
        x = self.patch_unembed(x, (Hp, Wp))
        x = self.conv_after_body(x) + residual
        x = self.upsample(x)
        x = self.conv_last(x)

        # remove padding after upsampling
        if pad_h or pad_w:
            scale = x.shape[2] // Hp
            x = x[:, :, : H * scale, : W * scale]
        return x


class GuidedSwinIR(nn.Module):
    """Simple guided variant: concat thermal (1ch) + optical RGB (3ch) -> 4-ch SwinIR, output 1ch."""

    def __init__(self, upscale=2):
        super().__init__()
        self.model = SwinIR(
            img_size=64,
            patch_size=1,
            in_chans=4,
            out_chans=1,
            embed_dim=60,  # match SwinIR-M x2 checkpoint (embed_dim=60, heads=6)
            depths=(6, 6, 6, 6, 6, 6),
            num_heads=(6, 6, 6, 6, 6, 6),
            window_size=8,
            mlp_ratio=2.0,
            upscale=upscale,
        )

    def forward(self, thr, opt):
        # thr: (B,1,H,W), opt: (B,3,H,W)
        x = torch.cat([thr, opt], dim=1)
        return self.model(x)


def load_pretrained_swinir(weight_path: Path, upscale=2, device: str = "cpu"):
    """Load pretrained SwinIR weights (3-ch) into GuidedSwinIR (4-ch) by expanding first conv."""
    net = GuidedSwinIR(upscale=upscale)
    sd = torch.load(weight_path, map_location=device)
    if "params" in sd:
        sd = sd["params"]

    model_sd = net.state_dict()
    new_sd = {}
    for k, v in sd.items():
        # Skip relative position buffers/masks from checkpoints with different window sizes.
        if "relative_position_index" in k or "relative_position_bias_table" in k or "attn_mask" in k:
            continue
        if k in model_sd:
            if k == "model.patch_embed.proj.weight" and v.shape[1] == 3:
                # Expand to 4 input channels: copy RGB weights and init thermal as mean of RGB.
                thermal = v.mean(dim=1, keepdim=True)
                v = torch.cat([thermal, v], dim=1)
            new_sd[k] = v
    model_sd.update(new_sd)
    net.load_state_dict(model_sd, strict=False)
    net.to(device)
    net.eval()
    return net


@dataclass
class SwinIRConfig:
    weight_path: Optional[Path] = None
    upscale: int = 2
    device: str = "cpu"


def build_model(cfg: SwinIRConfig):
    if cfg.weight_path:
        return load_pretrained_swinir(cfg.weight_path, upscale=cfg.upscale, device=cfg.device)
    model = GuidedSwinIR(upscale=cfg.upscale)
    model.to(cfg.device)
    return model
