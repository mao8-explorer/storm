#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, ReLU6
from .network_macros import MLPRegression, scale_to_base, scale_to_net
from ...util_file import get_weights_path, join_path


class ResidualBlock(nn.Module):
    def __init__(self, dim, act_fn=nn.SiLU, use_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.act1 = act_fn()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.act2 = act_fn()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(self.norm1(out))
        out = self.fc2(out)
        out = self.act2(self.norm2(out))
        return x + out

class ResidualMLP(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dim=256, num_blocks=3, use_nerf=True, dropout=0.1):
        super().__init__()
        self.use_nerf = use_nerf
        in_dim = input_dims * 2 if use_nerf else input_dims

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_layer = nn.Linear(hidden_dim, output_dims)

    def forward(self, x):
        if self.use_nerf:
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.dropout(x)
        return self.output_layer(x)


class RobotSelfCollisionNet(nn.Module):
    def __init__(self, n_joints=7, hidden_dim=256, num_blocks=3, dropout=0.1, use_nerf=True):
        super().__init__()
        self.model = ResidualMLP(
            input_dims=n_joints,
            output_dims=1,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            use_nerf=use_nerf,
            dropout=dropout
        )
        self.norm_dict = {}

    def load_weights(self, f_name, tensor_args):
        from storm_kit.util_file import join_path, get_weights_path
        try:
            chk = torch.load(join_path(get_weights_path(), f_name))
            self.model.load_state_dict(chk["model_state_dict"])
            self.norm_dict = chk["norm"]
            for k in self.norm_dict.keys():
                self.norm_dict[k]['mean'] = self.norm_dict[k]['mean'].to(**tensor_args)
                self.norm_dict[k]['std'] = self.norm_dict[k]['std'].to(**tensor_args)
        except Exception as e:
            print(f'WARNING: Weights not loaded: {e}')
        self.model = self.model.to(**tensor_args)
        self.model.eval()
        self.tensor_args = tensor_args

    def compute_signed_distance(self, q):
        with torch.no_grad():
            q_scaled = self.scale_to_net(q, 'x')
            pred = self.model(q_scaled)
            return self.scale_to_base(pred, 'y')

    def scale_to_net(self, x, key):
        return (x - self.norm_dict[key]['mean']) / (self.norm_dict[key]['std'] + 1e-6)

    def scale_to_base(self, x, key):
        return x * (self.norm_dict[key]['std'] + 1e-6) + self.norm_dict[key]['mean']