# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TensoRF Field"""


from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from tensorf_components.components import Encoding, Identity, SHEncoding

from tensorf_components.components import MLP
from tensorf_components.field_heads import RGBFieldHead




def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

class TensoRFField(nn.Module):
    """TensoRF Field"""

    def __init__(
        self,
        aabb: Tensor,
        # the aabb bounding box of the dataset
        feature_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for appearance encoding outputs
        direction_encoding: Encoding = Identity(in_dim=3),
        # the encoding method used for ray direction
        density_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene density
        color_encoding: Encoding = Identity(in_dim=3),
        # the tensor encoding method used for scene color
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        head_mlp_num_layers: int = 2,
        # number of layers for the MLP
        head_mlp_layer_width: int = 128,
        # layer width for the MLP
        use_sh: bool = False,
        # whether to use spherical harmonics as the feature decoding function
        sh_levels: int = 2,
        # number of levels to use for spherical harmonics
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.feature_encoding = feature_encoding
        self.direction_encoding = direction_encoding
        self.density_encoding = density_encoding
        self.color_encoding = color_encoding

        self.mlp_head = MLP(
            in_dim=appearance_dim + 3 + self.direction_encoding.get_out_dim() + self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        self.use_sh = use_sh

        if self.use_sh:
            self.sh = SHEncoding(sh_levels)
            self.B = nn.Linear(
                in_features=self.color_encoding.get_out_dim(), out_features=3 * self.sh.get_out_dim(), bias=False
            )
        else:
            self.B = nn.Linear(in_features=self.color_encoding.get_out_dim(), out_features=appearance_dim, bias=False)

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), activation=nn.Sigmoid())

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> Tensor:
        
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        
        density = self.density_encoding(pts)
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        # relu = torch.nn.ReLU()
        # density_enc = relu(density_enc)
        density = self.density_activation(density_enc).view(n_rays, n_samples)

        return density

    def get_outputs(self, pts: torch.Tensor, directions: torch.Tensor, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = directions
        positions = pts
        rgb_features = self.color_encoding(positions)
        rgb_features = self.B(rgb_features)

        if self.use_sh:
            sh_mult = self.sh(d)[:, :, None]
            rgb_sh = rgb_features.view(sh_mult.shape[0], sh_mult.shape[1], 3, sh_mult.shape[-1])
            rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
        else:
            d_encoded = self.direction_encoding(d)
            rgb_features_encoded = self.feature_encoding(rgb_features)

            out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))  # type: ignore
            rgb = self.field_output_rgb(out)

        return rgb

    def forward(
        self,
        pts: torch.Tensor,
        directions: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None):
        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)

        density = self.get_density(pts, timestamps)
        rgb = self.get_outputs(pts, directions, None)

        return {'rgb': rgb, 'density':density}
    
    def get_param_groups(self):
        param_groups = {}

        param_groups["fields"] = (
            list(self.mlp_head.parameters())
            + list(self.B.parameters())
            + list(self.field_output_rgb.parameters())
        )
        param_groups["encodings"] = list(self.color_encoding.parameters()) + list(
            self.density_encoding.parameters()
        )
        
        return param_groups
