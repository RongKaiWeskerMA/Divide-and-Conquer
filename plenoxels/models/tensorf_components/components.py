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

"""
Encoding functions
"""

import itertools
from abc import abstractmethod
from typing import Literal, Optional, Sequence, Tuple, Union, Set

import numpy as np
import torch
import torch.nn.functional as F
from typing import Float, Int, Shaped
from torch import Tensor, nn
import tinycudann as tcnn
from math import  components_from_spherical_harmonics, expected_sin
from external import TCNN_EXISTS


class Encoding(nn.Module):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @abstractmethod
    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(nn.Module):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: Shaped[Tensor, "*bs input_dim"]) -> Shaped[Tensor, "*bs output_dim"]:
        return in_tensor
    

class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical harmonic levels to encode.
    """

    def __init__(self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "torch") -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only supports 1 to 4 levels, requested {levels}")

        self.levels = levels

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            pass
        elif implementation == "tcnn":
            encoding_config = {
                "otype": "SphericalHarmonics",
                "degree": levels,
            }
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)
    
def activation_to_tcnn_string(activation: Union[nn.Module, None]) -> str:
    """Converts a torch.nn activation function to a string that can be used to
    initialize a TCNN activation function.

    Args:
        activation: torch.nn activation function
    Returns:
        str: TCNN activation function string
    """

    if isinstance(activation, nn.ReLU):
        return "ReLU"
    if isinstance(activation, nn.LeakyReLU):
        return "Leaky ReLU"
    if isinstance(activation, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(activation, nn.Softplus):
        return "Softplus"
    if isinstance(activation, nn.Tanh):
        return "Tanh"
    if isinstance(activation, type(None)):
        return "None"
    tcnn_documentation_url = "https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions"
    raise ValueError(
        f"TCNN activation {activation} not supported for now.\nSee {tcnn_documentation_url} for TCNN documentation."
    )


class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        if implementation == "torch":
            self.build_nn_modules()
        elif implementation == "tcnn" and not TCNN_EXISTS:
            print("tcnn not exist!!! please build")
            self.build_nn_modules()
        elif implementation == "tcnn":
            activation_str = activation_to_tcnn_string(activation)
            output_activation_str = activation_to_tcnn_string(out_activation)
            if layer_width in [16, 32, 64, 128]:
                network_config = {
                    "otype": "FullyFusedMLP",
                    "activation": activation_str,
                    "output_activation": output_activation_str,
                    "n_neurons": layer_width,
                    "n_hidden_layers": num_layers - 1,
                }
            # else:
            #     CONSOLE.line()
            #     CONSOLE.print("[bold yellow]WARNING: Using slower TCNN CutlassMLP instead of TCNN FullyFusedMLP")
            #     CONSOLE.print(
            #         "[bold yellow]Use layer width of 16, 32, 64, or 128 to use the faster TCNN FullyFusedMLP."
            #     )
            #     CONSOLE.line()
            #     network_config = {
            #         "otype": "CutlassMLP",
            #         "activation": activation_str,
            #         "output_activation": output_activation_str,
            #         "n_neurons": layer_width,
            #         "n_hidden_layers": num_layers - 1,
            #     }

            self.tcnn_encoding = tcnn.Network(
                n_input_dims=in_dim,
                n_output_dims=out_dim,
                network_config=network_config,
            )

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def pytorch_fwd(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)
    

class TensorVMEncoding(Encoding):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]
    line_coef: Float[Tensor, "3 num_components resolution 1"]

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        self.plane_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, resolution)))
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        plane_coord = plane_coord.view(3, -1, 1, 2).detach()
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        plane_features = F.grid_sample(self.plane_coef, plane_coord, align_corners=True)  # [3, Components, -1, 1]
        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = plane_features * line_features  # [3, Components, -1, 1]
        features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., 3 * Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )
        line_coef = F.interpolate(self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True)

        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.resolution = resolution


class NeRFEncoding(Encoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        implementation: Literal["tcnn", "torch"] = "torch",
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

        self.tcnn_encoding = None
        if implementation == "tcnn" and not TCNN_EXISTS:
            print('build tcnn')
        elif implementation == "tcnn":
            encoding_config = {"otype": "Frequency", "n_frequencies": num_frequencies}
            assert min_freq_exp == 0, "tcnn only supports min_freq_exp = 0"
            assert max_freq_exp == num_frequencies - 1, "tcnn only supports max_freq_exp = num_frequencies - 1"
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=in_dim,
                encoding_config=encoding_config,
            )

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def pytorch_fwd(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
        covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None,
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs

    def forward(
        self, in_tensor: Float[Tensor, "*bs input_dim"], covs: Optional[Float[Tensor, "*bs input_dim input_dim"]] = None
    ) -> Float[Tensor, "*bs output_dim"]:
        if self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor, covs)