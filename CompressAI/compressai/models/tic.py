import math
from click import prompt
import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import RSTB, MultistageMaskedConv2d, CausalAttentionModule
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from numpy import ceil

from .utils import conv, deconv, update_registered_buffers, Demultiplexer, Multiplexer

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class TIC(nn.Module):
    """Neural image compression framework from 
    Lu Ming and Guo, Peiyao and Shi, Huiqing and Cao, Chuntong and Ma, Zhan: 
    `"Transformer-based Image Compression" <https://arxiv.org/abs/2111.06707>`, (DCC 2022).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
        input_resolution (int): Just used for window partition decision
    """

    def __init__(self, N=128, M=192):
        super().__init__()

        depths = [2, 4, 6, 2, 2]
        num_heads = [4, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 4.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.2
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(128,128),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(64,64),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(32,32),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)

        self.h_a0 = conv(M, N, kernel_size=3, stride=1)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(16,16),
                         depth=depths[3],
                         num_heads=num_heads[3],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(8,8),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_a4 = conv(N, N, kernel_size=3, stride=2)

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s1 = RSTB(dim=N,
                         input_resolution=(8,8),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s3 = RSTB(dim=N,
                         input_resolution=(16,16),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         )
        self.h_s4 = conv(N, M*2, kernel_size=3, stride=1)
        
        self.g_s0 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s1 = RSTB(dim=N,
                        input_resolution=(32,32),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s3 = RSTB(dim=N,
                        input_resolution=(64,64),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s4 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s5 = RSTB(dim=N,
                        input_resolution=(128,128),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        )
        self.g_s6 = deconv(N, 3, kernel_size=5, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.context_prediction = CausalAttentionModule(M, M*2)   

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M*12//3, M*10//3, 1),
            nn.GELU(),
            nn.Conv2d(M*10//3, M*8//3, 1),
            nn.GELU(),
            nn.Conv2d(M*8//3, M*6//3, 1),
        )   

        self.apply(self._init_weights)   

    def g_a(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)
        x, _ = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_a2(x)
        x, _ = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_a4(x)
        x, _ = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_a6(x)
        return x

    def g_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.g_s0(x)
        x, _ = self.g_s1(x, (x_size[0]//8, x_size[1]//8))
        x = self.g_s2(x)
        x, _ = self.g_s3(x, (x_size[0]//4, x_size[1]//4))
        x = self.g_s4(x)
        x, _ = self.g_s5(x, (x_size[0]//2, x_size[1]//2))
        x = self.g_s6(x)
        return x

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//16, x_size[1]//16))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a4(x)
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x = self.h_s0(x)
        x, _ = self.h_s1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s2(x)
        x, _ = self.h_s3(x, (x_size[0]//16, x_size[1]//16))
        x = self.h_s4(x)
        return x
    
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size)
        z = self.h_a(y, x_size)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat, x_size)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, x_size)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # def hierarchical_forward(self, x):
    #     x_size = (x.shape[2], x.shape[3])
    #     y = self.g_a(x, x_)

    
    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        # pylint: enable=protected-access
        # print(cdf, cdf_lengths, offsets)
        y_strings = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            symbols_list = []
            indexes_list = []
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[
                        i : i + 1, :, h : h + kernel_size, w : w + kernel_size
                    ]
                    ctx_p = self.context_prediction(y_crop)
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i : i + 1, :, h : h + 1, w : w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[i : i + 1, :, 2 : 3, 2 : 3]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop - means_hat)
                    y_hat[i, :, h + padding, w + padding] = (y_q + means_hat)[
                        i, :, padding, padding
                    ]

                    symbols_list.extend(y_q[i, :, padding, padding].int().tolist())
                    indexes_list.extend(indexes[i, :].squeeze().int().tolist())

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )

            string = encoder.flush()
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        decoder = RansDecoder()

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for i, y_string in enumerate(strings[0]):
            decoder.set_stream(y_string)

            for h in range(y_height):
                for w in range(y_width):
                    # only perform the 5x5 convolution on a cropped tensor
                    # centered in (h, w)
                    y_crop = y_hat[
                        i : i + 1, :, h : h + kernel_size, w : w + kernel_size
                    ]
                    ctx_p = self.context_prediction(y_crop)
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i : i + 1, :, h : h + 1, w : w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p[i : i + 1, :, 2 : 3, 2 : 3]), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)

                    rv = decoder.decode_stream(
                        indexes[i, :].squeeze().int().tolist(),
                        cdf,
                        cdf_lengths,
                        offsets,
                    )
                    rv = torch.Tensor(rv).reshape(1, -1, 1, 1)

                    rv = self.gaussian_conditional._dequantize(rv, means_hat)

                    y_hat[
                        i,
                        :,
                        h + padding : h + padding + 1,
                        w + padding : w + padding + 1,
                    ] = rv
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        # pylint: enable=protected-access

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class Alignment(torch.nn.Module):
    """Image Alignment for model downsample requirement"""

    def __init__(self, divisor=64., mode='pad', padding_mode='replicate'):
        super().__init__()
        self.divisor = float(divisor)
        self.mode = mode
        self.padding_mode = padding_mode
        self._tmp_shape = None

    def extra_repr(self):
        s = 'divisor={divisor}, mode={mode}'
        if self.mode == 'pad':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    @staticmethod
    def _resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def _align(self, input):
        H, W = input.size()[-2:]
        H_ = int(ceil(H / self.divisor) * self.divisor)
        W_ = int(ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
            self._tmp_shape = None
            return input

        self._tmp_shape = input.size()
        if self.mode == 'pad':
            return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(H_, W_))

    def _resume(self, input, shape=None):
        if shape is not None:
            self._tmp_shape = shape
        if self._tmp_shape is None:
            return input

        if self.mode == 'pad':
            output = input[..., :self._tmp_shape[-2], :self._tmp_shape[-1]]
        elif self.mode == 'resize':
            output = self._resize(input, size=self._tmp_shape[-2:])

        return output

    def align(self, input):
        """align"""
        if input.dim() == 4:
            return self._align(input)

    def resume(self, input, shape=None):
        """resume"""
        if input.dim() == 4:
            return self._resume(input, shape)

    def forward(self, func, *args, **kwargs):
        pass

class TIC_hp(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, N=128, M=192, prompt_config=None, input_resolution=(256,256), in_channel=3):
        super().__init__()

        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint= False

        if prompt_config is not None:
            architect = prompt_config.ARCHITECT
        else:
            architect = None


        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[0],
                        num_heads=num_heads[0],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'encoder'] else None
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[1],
                        num_heads=num_heads[1],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'encoder'] else None
        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'encoder'] else None
        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'encoder'] else None
        )

        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[4],
                         num_heads=num_heads[4],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if (architect in ['both', 'encoder']) and prompt_config.HYPERPRIOR  else None
        )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[5],
                         num_heads=num_heads[5],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if (architect in ['both', 'encoder']) and prompt_config.HYPERPRIOR else None
        )

        depths = depths[::-1]
        num_heads = num_heads[::-1]
        self.h_s0 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//64, input_resolution[1]//64),
                         depth=depths[0],
                         num_heads=num_heads[0],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if (architect in ['both', 'decoder']) and prompt_config.HYPERPRIOR else None
        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(dim=N,
                         input_resolution=(input_resolution[0]//32, input_resolution[1]//32),
                         depth=depths[1],
                         num_heads=num_heads[1],
                         window_size=window_size//2,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:1]):sum(depths[:2])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if (architect in ['both', 'decoder']) and prompt_config.HYPERPRIOR else None
        )
        self.h_s3 = deconv(N, M*2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.g_s0 = RSTB(dim=M,
                        input_resolution=(input_resolution[0]//16, input_resolution[1]//16),
                        depth=depths[2],
                        num_heads=num_heads[2],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:2]):sum(depths[:3])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'decoder'] else None
        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s2 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//8, input_resolution[1]//8),
                        depth=depths[3],
                        num_heads=num_heads[3],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:3]):sum(depths[:4])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'decoder'] else None
        )
        self.g_s3 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s4 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//4, input_resolution[1]//4),
                        depth=depths[4],
                        num_heads=num_heads[4],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:4]):sum(depths[:5])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'decoder'] else None
        )
        self.g_s5 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s6 = RSTB(dim=N,
                        input_resolution=(input_resolution[0]//2, input_resolution[1]//2),
                        depth=depths[5],
                        num_heads=num_heads[5],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:5]):sum(depths[:6])],
                        norm_layer=norm_layer,
                        use_checkpoint=use_checkpoint,
                        prompt_config= prompt_config if architect in ['both', 'decoder'] else None
        )
        self.g_s7 = deconv(N, 3, kernel_size=5, stride=2)

        self.apply(self._init_weights)   

    def g_a(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)
        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)
        x = self.g_a2(x)
        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)
        x = self.g_a4(x)
        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)
        x = self.g_a6(x)
        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        return x, attns

    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x, attn = self.g_s0(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        x = self.g_s1(x)
        x, attn = self.g_s2(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)
        x = self.g_s3(x)
        x, attn = self.g_s4(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)
        x = self.g_s5(x)
        x, attn = self.g_s6(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)
        x = self.g_s7(x)
        return x, attns

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*16, x.shape[3]*16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//64, x_size[1]//64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2]*64, x.shape[3]*64)
        x, _ = self.h_s0(x, (x_size[0]//64, x_size[1]//64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0]//32, x_size[1]//32))
        x = self.h_s3(x)
        return x

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat, x_size)
        
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat, attns_s = self.g_s(y_hat, x_size)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "attn_a": attns_a,
            "attn_s": attns_s,
            'y_hat': y_hat
        }

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        x_size = (x.shape[2], x.shape[3])
        y, attns_a = self.g_a(x, x_size)
        z = self.h_a(y, x_size)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat, attns_s = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}