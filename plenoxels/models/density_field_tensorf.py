
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from tensorf_components.components import Encoding, Identity, SHEncoding

from tensorf_components.components import MLP
from tensorf_components.field_heads import RGBFieldHead




def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


class TensoRFDensityField(nn.Module):
    """TensoRF Field"""

    def __init__(
        self,
        aabb: Tensor,
        # the aabb bounding box of the dataset
        density_encoding: Encoding = Identity(in_dim=3),
        # the number of dimensions for the appearance embedding
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.density_encoding = density_encoding


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
        density = self.density_activation(density_enc).view(n_rays, n_samples)

        return density

    def forward(self, pts: torch.Tensor):
        return self.get_density(pts)

    def get_params(self):
        param_groups = {}
        param_groups["encodings"] = list(self.density_encoding.parameters())
        
        return param_groups