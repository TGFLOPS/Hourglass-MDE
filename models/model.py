# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# HourglassNeck by TGFLOPS
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init, ConvModule)
from models.swin_transformer_v2 import SwinTransformerV2
from deformable_attention import DeformableAttention2D

    
class HourglassNeck(nn.Module):
    def __init__(self, in_dim=1536, mid_dim=768, out_dim=1536):
        super(HourglassNeck, self).__init__()
        self.hourglass = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_dim, 384, kernel_size=1, bias=False),
            nn.ReLU(),
            DeformableAttention2D(dim=384, 
                                    dim_head=48, 
                                    heads=8, 
                                    dropout=0., 
                                    downsample_factor=3, 
                                    offset_scale=4, 
                                    offset_groups=None, 
                                    offset_kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(384, mid_dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_dim, out_dim, kernel_size=1, bias=False),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # print("before_neck : ", x[0].shape)
        fx = self.hourglass(x[0])  # F(x)
        # print("after_neck : ", fx.shape)
        out = fx + x[0]  # F(x) + x
        # print("after_fx+x[0]_neck : ", out.shape)
        out = self.relu(out)
        # print("final_neck : ", out.shape)
        return out


class GLPDepth(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth
        
        if 'tiny' in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif 'base' in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif 'large' in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif 'huge' in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        else:
            raise ValueError(args.backbone+" is not implemented, please add it in the models/model.py.")

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=args.depths,
            num_heads=num_heads,
            window_size=args.window_size,
            pretrain_window_size=args.pretrain_window_size,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_shift=args.use_shift,
        )


        self.encoder.init_weights(pretrained=args.pretrained)
        
        channels_in = embed_dim*8
        channels_out = embed_dim
        
        self.neck = HourglassNeck(in_dim=channels_in, mid_dim=768, out_dim=channels_in)

        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):                
        conv_feats = self.encoder(x)
        conv_feats = self.neck(conv_feats)
        out = self.decoder(conv_feats)
        out_depth = self.last_layer_depth(out)
        # print("after_last_layer : ", out_depth.shape)
        out_depth = torch.sigmoid(out_depth) * self.max_depth
        # print("after_sigmoid : ", out_depth.shape)

        return {'pred_d': out_depth}


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels
        
        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv, # 3
            args.num_filters, # [32,32,32]
            args.deconv_kernels, # [2,2,2]
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        # out = self.deconv_layers(conv_feats[0])
        out = self.deconv_layers(conv_feats)
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers): # 3
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

