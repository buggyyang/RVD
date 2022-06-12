import torch
from torch import nn
from .utils import exists, default
from .network_components import (
    Residual,
    SinusoidalPosEmb,
    Upsample,
    Downsample,
    PreNorm,
    Block,
    LinearAttention,
    get_backbone,
)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        context_dim_factor=1,
        dim_mults=(1, 1, 2, 2, 4, 4),
        channels=3,
        with_time_emb=True,
        backbone="resnet",  # convnext or resnet
    ):
        super().__init__()
        self.channels = channels
        self.context_dim_factor = context_dim_factor

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_in, dim_out, time_dim)),
                        get_backbone(backbone, (dim_out + int(dim_out * self.context_dim_factor), dim_out, time_dim)),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = get_backbone(backbone, (mid_dim, mid_dim, time_dim))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        get_backbone(backbone, (dim_out * 2, dim_in, time_dim)),
                        get_backbone(backbone, (dim_in, dim_in, time_dim)),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(Block(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def encode(self, x, t, context):
        h = []
        for idx, (backbone, backbone2, attn, downsample) in enumerate(self.downs):
            x = backbone(x, t)
            x = backbone2(torch.cat((x, context[idx]), dim=1), t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        return x, h

    def decode(self, x, h, t):
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for backbone, backbone2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = backbone(x, t)
            x = backbone2(x, t)
            x = attn(x)
            x = upsample(x)
        return self.final_conv(x)

    def forward(self, x, time=None, context=None):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        x, h = self.encode(x, t, context)
        return self.decode(x, h, t)
