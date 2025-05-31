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

from basicsr.archs.arch_util import trunc_normal_
import numpy as np



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
    def __init__(self, dim=180, num_heads=6, window_size=12, num_gs_seed=2304):
        super(WindowCrossAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_gs_seed = num_gs_seed
        self.num_gs_seed_sqrt = int(math.sqrt(num_gs_seed))

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * self.num_gs_seed_sqrt - 1) * (2 * self.num_gs_seed_sqrt - 1), num_heads))
        # trunc_normal_(self.relative_position_bias_table, std=.02)

        # # get pair-wise relative position index for each token inside the window
        coords_source = (np.indices((self.num_gs_seed_sqrt, self.num_gs_seed_sqrt)) + 0.5) * self.window_size
        coords_tgt = (np.indices((self.window_size, self.window_size)) + 0.5) * self.num_gs_seed_sqrt
        coords_delta = coords_source.reshape(2,-1)[:, :, np.newaxis] - coords_tgt.reshape(2, -1)[:, np.newaxis, :]
        shape = coords_delta.shape
        indexed_coords_delta = coords_delta.flatten()
        indexed_coords_delta = list(map(list(set(sorted(indexed_coords_delta))).index, indexed_coords_delta))
        indexed_coords_delta = np.array(indexed_coords_delta).reshape(shape)

        indexed_coords_delta[0] *= indexed_coords_delta.max()
        indexed_coords_delta = torch.from_numpy(indexed_coords_delta.sum(0))
        self.register_buffer('relative_position_index', indexed_coords_delta)
        # assign closed feature bigger initial weight
        relative_position_bias_table = torch.zeros((2 * max(self.num_gs_seed_sqrt, window_size) - 1) * (2 * max(self.num_gs_seed_sqrt, window_size) - 1), num_heads)
        for source_pix, closest_tgt_pix in enumerate((coords_delta**2).sum(0).argmin(1)):
            rel_pos_idx = indexed_coords_delta[source_pix][closest_tgt_pix]
            relative_position_bias_table[rel_pos_idx] = 2
        self.relative_position_bias_table = nn.Parameter(relative_position_bias_table)
        trunc_normal_(self.relative_position_bias_table, std=.02)


        self.qhead = nn.Linear(dim, dim, bias=True)
        self.khead = nn.Linear(dim, dim, bias=True)
        self.vhead = nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

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

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # b_, num_heads, num_gs, n

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # print(f"attn.norm(), {attn[0].norm()} \t relative_position_bias.norm(): {relative_position_bias.norm()}")
        attn = attn + relative_position_bias.unsqueeze(0) #  * 100
        attn = self.softmax(attn)
        # always focus self
        # attn = attn * 0 +torch.eye(144, device=attn.device)[None,None]
        x = (attn @ v).transpose(1, 2).reshape(b_, num_gs, c)
        x = self.proj(x)

        return x


class WindowCrossAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, window_size=12, shift_size=0, num_gs_seed=2308):
        super(WindowCrossAttnLayer, self).__init__()

        self.gs_cross_attn_scale = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.shift_size = shift_size
        self.window_size = window_size

        self.window_cross_attn = WindowCrossAttn(dim=dim, num_heads=num_heads, window_size=window_size,
                                                         num_gs_seed=num_gs_seed)
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
    def __init__(self, dim=180, window_size=12, num_heads=6, num_layers=4, num_gs_seed=2308):
        super(WindowCrossAttnBlock, self).__init__()

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
                num_gs_seed=num_gs_seed) for i in range(num_layers)
        ])

    def forward(self, x, query_pos, feat, scale_embedding):
        resi = x
        x = self.norm(x)
        for block in self.blocks:
            x = block(x, query_pos, feat, scale_embedding)
        x = self.mlp(x)
        x = resi + x
        return x


class GSSelfAttn(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt = 12):
        super(GSSelfAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_gs_seed_sqrt = num_gs_seed_sqrt

        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.proj = nn.Linear(dim, dim)
       
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.num_gs_seed_sqrt - 1) * (2 *self.num_gs_seed_sqrt - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.num_gs_seed_sqrt)
        coords_w = torch.arange(self.num_gs_seed_sqrt)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.num_gs_seed_sqrt - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.num_gs_seed_sqrt - 1
        relative_coords[:, :, 0] *= 2 * self.num_gs_seed_sqrt - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

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

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # b_, num_heads, num_gs, n

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, self.num_gs_seed_sqrt * self.num_gs_seed_sqrt, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)

        attn = (attn @ v).transpose(1, 2).reshape(b_, num_gs, c)
        attn = self.proj(attn)

        return attn


class GSSelfAttnLayer(nn.Module):
    def __init__(self, dim=180, num_heads=6, num_gs_seed_sqrt = 12, shift_size = 0):
        super(GSSelfAttnLayer, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.gs_self_attn = GSSelfAttn(dim = dim, num_heads = num_heads, num_gs_seed_sqrt = num_gs_seed_sqrt)

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
    def __init__(self, dim=180, num_heads=6, num_selfattn_layers=4, num_gs_seed_sqrt = 12):
        super(GSSelfAttnBlock, self).__init__()

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
                shift_size=0 if i % 2 == 0 else num_gs_seed_sqrt // 2
            ) for i in range(num_selfattn_layers)
        ])

    def forward(self, gs, pos, h_count, w_count, scale_embedding):
        resi = gs
        gs = self.norm(gs)
        for block in self.blocks:
            gs = block(gs, pos, h_count, w_count, scale_embedding)
        gs = self.mlp(gs)
        gs = gs + resi
        return gs

@ARCH_REGISTRY.register()
class Fea2GS(nn.Module):
    def __init__(self, inchannel=64, channel=180, num_heads=6, num_crossattn_blocks=1, num_crossattn_layers=2, num_selfattn_blocks = 6, num_selfattn_layers = 6,
                 num_gs_seed=144, gs_up_factor=1.0, window_size=12, img_range=1.0, shuffle_scale1 = 2, shuffle_scale2 = 2, use_checkpoint = False):
        super(Fea2GS, self).__init__()
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
                                 num_gs_seed=num_gs_seed) for i in range(num_crossattn_blocks)
        ])

        self.gs_selfattn_blocks = nn.ModuleList([
            GSSelfAttnBlock(dim=channel,
                            num_heads=num_heads,
                            num_selfattn_layers=num_selfattn_layers,
                            num_gs_seed_sqrt=self.num_gs_seed_sqrt
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
                query = checkpoint(block, query, query_pos, feat, scale_embedding)
            else:
                query = block(query, query_pos, feat, scale_embedding)  # b*h_count*w_count, num_gs_seed, channel

        resi = query
        for block in self.gs_selfattn_blocks:
            if self.use_checkpoint:
                query = checkpoint(block, query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
            else:
                query = block(query, query_pos, h // self.window_size, w // self.window_size, scale_embedding)
        query = query + resi

        query = rearrange(query, '(b m n) (h w) c -> b c (m h) (n w)', m=h // self.window_size, n=w // self.window_size,
                          h=self.num_gs_seed_sqrt)
        ### We make the number of GS = 16*LR_size in the following, through an simple upsample operation
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
    srcs = torch.randn(6, 64, 48, 48).cuda()
    scale = torch.randn(6).cuda()
    decoder = Fea2GS().cuda()
    import time
    for i in range(10):
        torch.cuda.synchronize()
        time1 = time.time()

        y = decoder(srcs, scale)
        torch.cuda.synchronize()
        time2 = time.time()
        print(f"decoder time is {time2 - time1}")
        print(y.shape)
    pass