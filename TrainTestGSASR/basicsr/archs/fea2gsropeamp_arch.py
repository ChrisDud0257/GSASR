from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import copy
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_, kaiming_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from functools import partial
from typing import Any, Optional, Tuple

from basicsr.archs.arch_util import trunc_normal_
import numpy as np

def init_t_xy(end_x: int, end_y: int, zero_center=False):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    
    return t_x, t_y

def init_random_2d_freqs(head_dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    theta = theta
    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_cis(freqs, t_x, t_y):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
        
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    # assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    # print(f"freqs_cis shape is {freqs_cis.shape}, x shape is {x.shape}")
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # print(f"xq shape is {xq.shape}, xq.shape[:-1] is {xq.shape[:-1]}")
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # print(f"xq_ shape is {xq_.shape}")
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def apply_rotary_emb_single(x, freqs_cis):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    seq_len = x_.shape[2]
    freqs_cis = freqs_cis[:, :seq_len, :]
    freqs_cis = freqs_cis.unsqueeze(0).expand_as(x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x).to(x.device)

def window_partition(x, window_size):
    # x is the feature from net_g
    b, c, h, w = x.shape
    windows = rearrange(x, 'b c (h_count dh) (w_count dw) -> (b h_count w_count) (dh dw) c', dh=window_size,
                        dw=window_size)
    # h_count = h // window_size
    # w_count = w // window_size
    # windows = x.reshape(b,c,h_count, window_size, w_count, window_size)
    # windows = windows.permute(0,1,2,4,3,5) #b,c,h_count,w_count,window_size,window_size
    # windows = windows.reshape(b,c,h_count*w_count, window_size * window_size)
    # windows = windows.permute(0,2,3,1) #b,h_count*w_count, window_size*window_size,c
    # windows = windows.reshape(-1, window_size*window_size, c)

    return windows


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.ReLU):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowCrossAttn(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=12, num_gs_seed=2304, rope_mixed = True, rope_theta = 10.0):
        super(WindowCrossAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_gs_seed = num_gs_seed
        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))


        self.rope_mixed = rope_mixed
       
        t_x, t_y = init_t_xy(end_x=max(self.num_gs_seed_sqrt, self.window_size), end_y=max(self.num_gs_seed_sqrt, self.window_size))
        self.register_buffer('rope_t_x', t_x)
        self.register_buffer('rope_t_y', t_y)

        freqs = init_random_2d_freqs(
            head_dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta, 
            rotate=self.rope_mixed
        )
        if self.rope_mixed:
            self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer('rope_freqs', freqs)
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
            self.rope_freqs_cis = freqs_cis

        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)


    def forward(self, gs, feat):
        # gs shape: b*h_count*w_count, num_gs, c    the input gs here should already include pos embedding and scale embedding
        # feat shape: b*h_count*w_count, dh*dw, c    dh=dw=window_size
        b_, num_gs, c = gs.shape
        b_, n, c = feat.shape

        q = self.qhead(gs)  # b_, num_gs_, c
        q = q.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        k = self.khead(feat)  # b_, n_, c
        k = k.reshape(b_, n, self.num_heads, c // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        v = self.vhead(feat)  # b_, n_, c
        v = v.reshape(b_, n, self.num_heads, c // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        ###### Apply rotary position embedding
        if self.rope_mixed:
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
        else:
            freqs_cis = self.rope_freqs_cis.to(gs.device)
        q = apply_rotary_emb_single(q, freqs_cis)
        k = apply_rotary_emb_single(k, freqs_cis)
        #########

        attn = F.scaled_dot_product_attention(q, k, v)

        x = attn.transpose(1, 2).reshape(b_, num_gs, c)

        x = self.proj(x)

        return x


class WindowCrossAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=12, shift_size=0, num_gs_seed=2308, rope_mixed = True, rope_theta = 10.0):
        super(WindowCrossAttnLayer, self).__init__()

        self.gs_cross_attn_scale = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.shift_size = shift_size
        self.window_size = window_size

        self.window_cross_attn = WindowCrossAttn(dim=dim, num_heads=num_heads, window_size=window_size,
                                                         num_gs_seed=num_gs_seed, rope_mixed = rope_mixed, rope_theta = rope_theta)
        self.mlp_crossattn_scale = MLP(in_features=dim, hidden_features=dim, out_features=dim)
        self.mlp_crossattn_feature = MLP(in_features=dim, hidden_features=dim, out_features=dim)

    def forward(self, x, query_pos, feat, scale_embedding):
        # gs shape: b*h_count*w_count, num_gs, c
        # query_pos shape: b*h_count*w_count, num_gs, c
        # feat shape: b,c,h,w
        # scale_embedding shape: b*h_count*w_count, 1, c

        ###GS cross attn with scale embedding
        resi = x
        x = self.norm1(x)
        # print(f"x: {x.shape} {x.device}, query_pos: {query_pos.shape}, {query_pos.device}, scale_embedding: {scale_embedding.shape}, {scale_embedding.device}")
        x, _ = self.gs_cross_attn_scale(with_pos_embed(x, query_pos), scale_embedding, scale_embedding)
        x = resi + x

        ###FFN
        resi = x
        x = self.norm2(x)
        x = self.mlp_crossattn_scale(x)
        x = resi + x

        ###cross attention for Q,K,V
        resi = x
        x = self.norm3(x)
        if self.shift_size > 0:
            shift_feat = torch.roll(feat, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shift_feat = feat
        shift_feat = window_partition(shift_feat, self.window_size)  # b*h_count*w_count, dh*dw, c  dh=dw=window_size
        x = self.window_cross_attn(with_pos_embed(x, query_pos),
                                   shift_feat)  # b*h_count*w_count, num_gs, c  dh=dw=window_size
        x = resi + x

        ###FFN
        resi = x
        x = self.norm4(x)
        x = self.mlp_crossattn_feature(x)
        x = resi + x

        return x


class WindowCrossAttnBlock(nn.Module):
    def __init__(self, dim=180, window_size=12, num_heads=6, num_layers=4, num_gs_seed=230, rope_mixed = True, rope_theta = 10.0):
        super(WindowCrossAttnBlock, self).__init__()

        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            WindowCrossAttnLayer(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                num_gs_seed=num_gs_seed,
                rope_mixed = rope_mixed, rope_theta = rope_theta) for i in range(num_layers)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, query_pos, feat, scale_embedding, h_count, w_count):
        resi = x
        x = self.norm(x)
        for block in self.blocks:
            x = block(x, query_pos, feat, scale_embedding)
        x = self.mlp(x)

        x = rearrange(x, '(b m n) (h w) c -> b c (m h) (n w)', m=h_count, n=w_count, h=self.num_gs_seed_sqrt)
        x = self.conv(x)
        x = rearrange(x, 'b c (m h) (n w) -> (b m n) (h w) c', m=h_count, n=w_count, h=self.num_gs_seed_sqrt)

        x = resi + x
        return x


class GSSelfAttn(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt = 12, rope_mixed = True, rope_theta=10.0):
        super(GSSelfAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_gs_seed_sqrt = num_gs_seed_sqrt

        self.proj = nn.Linear(dim, dim)
        self.rope_mixed = rope_mixed
       
        t_x, t_y = init_t_xy(end_x=self.num_gs_seed_sqrt, end_y=self.num_gs_seed_sqrt)
        self.register_buffer('rope_t_x', t_x)
        self.register_buffer('rope_t_y', t_y)

        freqs = init_random_2d_freqs(
            head_dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta, 
            rotate=self.rope_mixed
        )
        if self.rope_mixed:
            self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            self.register_buffer('rope_freqs', freqs)
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
            self.rope_freqs_cis = freqs_cis

        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)

    def forward(self, gs):
        # gs shape: b*h_count*w_count, num_gs, c  
        # pos shape: b*h_count*w_count, num_gs, c
        b_, num_gs, c = gs.shape

        q = self.qhead(gs)
        q = q.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        k = self.khead(gs)
        k = k.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        k = k.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        v = self.vhead(gs)
        v = v.reshape(b_, num_gs, self.num_heads, c // self.num_heads)
        v = v.permute(0, 2, 1, 3)  # b_, num_heads, n, c // num_heads

        ###### Apply rotary position embedding
        if self.rope_mixed:
            freqs_cis = compute_cis(self.rope_freqs, self.rope_t_x, self.rope_t_y)
        else:
            freqs_cis = self.rope_freqs_cis.to(gs.device)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        #########

        attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).reshape(b_, num_gs, c)


        attn = self.proj(attn)

        return attn


class GSSelfAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt = 12, shift_size = 0, rope_mixed = True, rope_theta=10.0):
        super(GSSelfAttnLayer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.gs_self_attn = GSSelfAttn(dim = dim, num_heads = num_heads, num_gs_seed_sqrt = num_gs_seed_sqrt, rope_mixed = rope_mixed, rope_theta=rope_theta)

        self.mlp_selfattn = MLP(in_features=dim, hidden_features=dim, out_features=dim)

        self.num_gs_seed_sqrt = num_gs_seed_sqrt
        self.shift_size = shift_size

        self.gs_cross_attn_scale = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.mlp_crossattn = MLP(in_features=dim, hidden_features=dim, out_features=dim)

    def forward(self, gs, pos, h_count, w_count, scale_embedding):
        # gs shape:b*h_count*w_count, num_gs_seed, channel
        # pos shape: b*h_count*w_count, num_gs_seed, channel
        # scale_embedding shape: b*h_count*w_count, 1, channel

        # gs cross attn with scale_embedding
        resi = gs
        gs = self.norm3(gs)
        gs, _ = self.gs_cross_attn_scale(with_pos_embed(gs, pos), scale_embedding, scale_embedding)
        gs = gs + resi

        # FFN
        resi = gs
        gs = self.norm4(gs)
        gs = self.mlp_crossattn(gs)
        gs = gs + resi

        resi = gs
        gs = self.norm1(gs)

        #### shift gs
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count, h=self.num_gs_seed_sqrt, w = self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count, h=self.num_gs_seed_sqrt, w = self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        #### gs self attention
        gs = self.gs_self_attn(shift_gs)

        #### shift gs back
        if self.shift_size > 0:
            shift_gs = rearrange(gs, '(b m n) (h w) c -> b (m h) (n w) c', m=h_count, n=w_count, h=self.num_gs_seed_sqrt, w = self.num_gs_seed_sqrt)
            shift_gs = torch.roll(shift_gs, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            shift_gs = rearrange(shift_gs, 'b (m h) (n w) c -> (b m n) (h w) c', m=h_count, n=w_count, h=self.num_gs_seed_sqrt, w = self.num_gs_seed_sqrt)
        else:
            shift_gs = gs

        gs = shift_gs + resi

        #FFN
        resi = gs
        gs = self.norm2(gs)
        gs = self.mlp_selfattn(gs)
        gs = gs + resi
        return gs


class GSSelfAttnBlock(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_selfattn_layers=4, num_gs_seed_sqrt = 12, rope_mixed = True, rope_theta=10.0):
        super(GSSelfAttnBlock, self).__init__()
        self.num_gs_seed_sqrt = num_gs_seed_sqrt

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            GSSelfAttnLayer(
                dim = dim,
                num_heads = num_heads,
                num_gs_seed_sqrt=num_gs_seed_sqrt,
                shift_size=0 if i % 2 == 0 else num_gs_seed_sqrt // 2,
                rope_mixed = rope_mixed, rope_theta=rope_theta
            ) for i in range(num_selfattn_layers)
        ])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, gs, pos, h_count, w_count, scale_embedding):
        resi = gs
        gs = self.norm(gs)
        for block in self.blocks:
            gs = block(gs, pos, h_count, w_count, scale_embedding)

        gs = self.mlp(gs)
        gs = rearrange(gs, '(b m n) (h w) c -> b c (m h) (n w)', m=h_count, n=w_count, h=self.num_gs_seed_sqrt)
        gs = self.conv(gs)
        gs = rearrange(gs, 'b c (m h) (n w) -> (b m n) (h w) c', m=h_count, n=w_count, h=self.num_gs_seed_sqrt)
        gs = gs + resi
        return gs


@ARCH_REGISTRY.register()
class Fea2GS_ROPE_AMP(nn.Module):
    def __init__(self, inchannel=64, channel=180, num_heads=6, num_crossattn_blocks=1, num_crossattn_layers=2, num_selfattn_blocks = 4, num_selfattn_layers = 6,
                 num_gs_seed=144, gs_up_factor=1.0, window_size=12, img_range=1.0, shuffle_scale1 = 2, shuffle_scale2 = 2, use_checkpoint = False,
                 rope_mixed = True, rope_theta = 10.0):
        """
        Args:
            gs_repeat_factor: the ratio of gs embedding number and pixel number along  width&height,  will generate
            (h * gs_repeat_factor) * (w * gs_repeat_factor) gs embedding, higher values means repeat more gs embedding.
            gs_up_factor: how many 2d gaussian are generated by one gasussian embedding.
        """
        super(Fea2GS_ROPE_AMP, self).__init__()
        self.channel = channel
        self.nhead = num_heads
        self.gs_up_factor = gs_up_factor
        self.num_gs_seed = num_gs_seed
        self.window_size = window_size
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint

        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))
        self.gs_up_factor_sqrt = int(math.sqrt(gs_up_factor))

        self.shuffle_scale1 = shuffle_scale1
        self.shuffle_scale2 = shuffle_scale2

        # shared gaussian embedding and its pos embedding
        self.gs_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.randn(self.num_gs_seed, channel), requires_grad=True)

        self.img_feat_proj = nn.Sequential(
            nn.Conv2d(inchannel, channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )

        self.window_crossattn_blocks = nn.ModuleList([
            WindowCrossAttnBlock(dim=channel,
                                 window_size=window_size,
                                 num_heads=num_heads,
                                 num_layers=num_crossattn_layers,
                                 num_gs_seed=num_gs_seed, rope_mixed = rope_mixed, rope_theta = rope_theta) for i in range(num_crossattn_blocks)
        ])

        self.gs_selfattn_blocks = nn.ModuleList([
            GSSelfAttnBlock(dim=channel,
                            num_heads=num_heads,
                            num_selfattn_layers=num_selfattn_layers,
                            num_gs_seed_sqrt=self.num_gs_seed_sqrt,
                            rope_mixed = rope_mixed, rope_theta=rope_theta
                            ) for i in range(num_selfattn_blocks)
        ])

        # GS sigma_x, sigma_y
        self.mlp_block_sigma = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        # GS rho
        self.mlp_block_rho = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS alpha
        self.mlp_block_alpha = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(1 * gs_up_factor))
        )

        # GS RGB values
        self.mlp_block_rgb = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(3 * gs_up_factor))
        )

        # GS mean_x, mean_y
        self.mlp_block_mean = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(),
            nn.Linear(channel, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, int(2 * gs_up_factor))
        )

        self.scale_mlp = nn.Sequential(
            nn.Linear(1, channel * 4),
            nn.ReLU(),
            nn.Linear(channel * 4, channel)
        )

        self.UPNet = nn.Sequential(
            nn.Conv2d(channel, channel * self.shuffle_scale1 * self.shuffle_scale1, 3, 1, 1),
            nn.PixelShuffle(self.shuffle_scale1),
            nn.Conv2d(channel, channel * self.shuffle_scale2 * self.shuffle_scale2, 3, 1, 1),
            nn.PixelShuffle(self.shuffle_scale2)
        )

        self.conv_final = nn.Conv2d(channel, channel, 3, 1, 1)

    @staticmethod
    def get_N_reference_points(h, w, device='cuda'):
        # step_y = 1/(h+1)
        # step_x = 1/(w+1)
        step_y = 1 / h
        step_x = 1 / w
        ref_y, ref_x = torch.meshgrid(torch.linspace(step_y / 2, 1 - step_y / 2, h, dtype=torch.float32, device=device),
                                      torch.linspace(step_x / 2, 1 - step_x / 2, w, dtype=torch.float32, device=device))
        reference_points = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = reference_points[None, :, None]
        return reference_points

    def forward(self, srcs, scale):
        '''
        using deformable detr decoder for cross attention
        Args:
            query: (batch_size, num_query, dim)
            query_pos: (batch_size, num_query, dim)
            srcs: (batch_size, dim, h1, w1)
        '''
        b, c, h, w = srcs.shape  ###srcs is pad to the size that could be divided by window_size
        query = self.gs_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (w // self.window_size),
                                                                   1, 1)  # b, h_count*w_count, num_gs_seed, channel
        query = query.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                              self.channel)  # b*h_count*w_count, num_gs_seed, channel

        scale = 1 / scale
        scale = scale.unsqueeze(1)  # b*1
        scale_embedding = self.scale_mlp(scale)  # b*channel
        scale_embedding = scale_embedding.unsqueeze(1).unsqueeze(2).repeat(1, (h // self.window_size) * (
                    w // self.window_size), self.num_gs_seed, 1)  # b, h_count*w_count, num_gs_seed, channel
        scale_embedding = scale_embedding.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel) # b*h_count*w_count, num_gs_seed, channel

        query_pos = self.pos_embedding.unsqueeze(0).unsqueeze(1).repeat(b, (h // self.window_size) * (
                    w // self.window_size), 1, 1)  # b, h_count*w_count, num_gs_seed, channel

        feat = self.img_feat_proj(srcs)  # b*channel*h*w

        query_pos = query_pos.reshape(b * (h // self.window_size) * (w // self.window_size), -1,
                                      self.channel)  # b*h_count*w_count, num_gs_seed, channel

        for block in self.window_crossattn_blocks:
            if self.use_checkpoint:
                query = checkpoint(block, query, query_pos, feat, scale_embedding, h // self.window_size, w // self.window_size)
            else:
                query = block(query, query_pos, feat, scale_embedding, h // self.window_size, w // self.window_size)  # b*h_count*w_count, num_gs_seed, channel

        resi = query
        for block in self.gs_selfattn_blocks:
            if self.use_checkpoint:
                query = checkpoint(block, query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
            else:
                query = block(query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
        

        query = rearrange(query, '(b m n) (h w) c -> b c (m h) (n w)', m=h // self.window_size, n=w // self.window_size,
                          h=self.num_gs_seed_sqrt)
        query = self.conv_final(query)


        resi = rearrange(resi, '(b m n) (h w) c -> b c (m h) (n w)', m=h // self.window_size, n=w // self.window_size,
                          h=self.num_gs_seed_sqrt)

        query = query + resi
        query = self.UPNet(query)
        query = query.permute(0,2,3,1)

        # query = rearrange(query, '(b m n) (h w) c -> b m h n w c', m=h // self.window_size, n=w // self.window_size,
        #                   h=self.num_gs_seed_sqrt)

        query_sigma = self.mlp_block_sigma(query).reshape(b, -1, 2)
        query_rho = self.mlp_block_rho(query).reshape(b, -1, 1)
        query_alpha = self.mlp_block_alpha(query).reshape(b, -1, 1)
        query_rgb = self.mlp_block_rgb(query).reshape(b, -1, 3)
        query_mean = self.mlp_block_mean(query).reshape(b, -1, 2)

        query_mean = query_mean / torch.tensor(
            [self.num_gs_seed_sqrt * (w // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2, 
            self.num_gs_seed_sqrt * (h // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2])[
            None, None].to(query_mean.device)  # b, h_count*w_count*num_gs_seed, 2

        reference_offset = self.get_N_reference_points(self.num_gs_seed_sqrt * (h // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2,
                                                       self.num_gs_seed_sqrt * (w // self.window_size) * self.shuffle_scale1 * self.shuffle_scale2, srcs.device)
        query_mean = query_mean + reference_offset.reshape(1, -1, 2)

        query = torch.cat([query_sigma, query_rho, query_alpha, query_rgb, query_mean],
                          dim=-1)  # b, h_count*w_count*num_gs_seed, 9

        return query


if __name__ == '__main__':
    srcs = torch.randn(6, 64, 64, 64, requires_grad = True).cuda()
    scale = torch.randn(6).cuda()
    decoder = Fea2GS_ROPE_AMP(inchannel=64, channel=192, num_heads=6, 
                            num_crossattn_blocks=1, num_crossattn_layers=2, 
                            num_selfattn_blocks = 6, num_selfattn_layers = 6,
                            num_gs_seed=256, gs_up_factor=1.0, window_size=16, 
                            img_range=1.0, shuffle_scale1 = 2, shuffle_scale2 = 2).cuda()
    import time
    
    for i in range(10):
        torch.cuda.synchronize()
        time1 = time.time()
        # with torch.autocast(device_type = 'cuda'):
        y = decoder(srcs, scale)
        torch.cuda.synchronize()
        time2 = time.time()
        print(f"decoder time is {time2 - time1}")
        print(y.shape)

        torch.cuda.synchronize()
        time3 = time.time()
        y.sum().backward()
        torch.cuda.synchronize()
        time4 = time.time()
        print(f"backward time is {time4 - time3}")
            
  