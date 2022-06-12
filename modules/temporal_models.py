from numpy import identity
import torch.nn as nn
import torch.nn.functional as F
import torch
import kornia
import math
from .network_components import (
    LayerNorm,
    ResnetBlock,
    Upsample,
    Downsample,
    ConvGRUCell,
    Block,
    Residual,
    PreNorm,
    LinearAttention,
    SinusoidalPosEmb,
    get_backbone,
)


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.exp().clamp(math.sqrt(2), 20 * math.sqrt(2))


class SimpleHistoryNet(nn.Module):
    def __init__(
        self,
        dim,  # must be the same as main net
        dim_mults=(1, 2, 3, 4),
        channels=3,
        context_mode="residual",
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        assert context_mode in ["residual"]
        self.context_mode = context_mode
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mu = nn.Conv2d(dim, channels, 3, 1, 1)
        self.sigma = (
            nn.Sequential(nn.Conv2d(dim, channels, 3, 1, 1), Exp())
            if context_mode in ["transform"]
            else None
        )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        
        self.mid = ConvGRUCell(dim_out, dim_out, 3, n_layer=2)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out, dim_in)),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

    def init_state(self, shape):
        temp_shape = list(shape)
        temp_shape[-2] //= 2 ** (len(self.dim_mults) - 1)
        temp_shape[-1] //= 2 ** (len(self.dim_mults) - 1)
        self.mid.init_hidden(temp_shape)

    def forward(self, x):
        for idx, (resnet, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = downsample(x)
        x = self.mid(x)
        for idx, (resnet, upsample) in enumerate(self.ups):
            x = resnet(x)
            x = upsample(x)
        mu = self.mu(x)
        sigma = self.sigma(x) if self.context_mode in ["transform"] else torch.ones_like(mu)
        return (mu.clamp(-1, 1), sigma)


class HistoryNet(nn.Module):
    def __init__(
        self,
        dim,  # must be the same as main net
        dim_mults=(1, 2, 3, 4),
        channels=3,
        context_mode="residual",
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        assert context_mode in ["residual"]
        self.context_mode = context_mode
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.mu = nn.Conv2d(dim, channels, 3, 1, 1)
        self.sigma = (
            nn.Sequential(nn.Conv2d(dim, channels, 3, 1, 1), Exp())
            if context_mode in ["transform"]
            else None
        )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        ConvGRUCell(dim_out, dim_out, 3, n_layer=1),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        
        # self.mid = get_backbone(backbone, (dim_in[-1], dim_in[-1]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out if ind == 0 else dim_out*2, dim_in)),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

    def init_state(self, shape):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape)

    def forward(self, x):
        input_frame = x
        h = []
        for idx, (resnet, gru, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = gru(x)
            if idx != (len(self.downs) - 1):
                h.append(x)
            x = downsample(x)
        # x = self.mid(x)
        for idx, (resnet, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1) if idx != 0 else x
            x = resnet(x)
            x = upsample(x)
        mu = self.mu(x)
        sigma = self.sigma(x) if self.context_mode in ["transform"] else torch.ones_like(mu)
        return (mu.clamp(-1, 1), sigma)


class CondNet(nn.Module):
    def __init__(
        self,
        dim,  # must be the same as main net
        dim_mults=(1, 1, 2, 2, 4, 4),  # must be the same as main net
        channels=3,
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.dim_mults = dim_mults
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out)),
                        ConvGRUCell(dim_out, dim_out, 3, n_layer=1),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

    def init_state(self, shape):
        for i, ml in enumerate(self.downs):
            temp_shape = list(shape)
            temp_shape[-2] //= 2 ** i
            temp_shape[-1] //= 2 ** i
            ml[1].init_hidden(temp_shape)

    def forward(self, x):
        context = []
        for i, (resnet, conv, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = conv(x)
            context.append(x)
            x = downsample(x)
        return context
    