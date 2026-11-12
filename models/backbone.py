"""PP-LiteSeg/STDC1 from zh320 (Apache-2.0), original author zh320,
2023/07/15, arXiv:2204.02681. Modified: retain only frozen DGPR dependencies;
import paths isolated, registry/unused baseline decoder omitted.
Definitions and state names unchanged. See README.md and LICENSE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_ops import conv3x3,ConvBNAct

class Encoder(nn.Module):
    def __init__(self, in_channels, encoder_channels, encoder_type, act_type):
        super().__init__()
        encoder_hub = {'stdc1':STDCBackbone, 'stdc2':STDCBackbone}
        if encoder_type not in encoder_hub.keys():
            raise ValueError(f'Unsupport encoder type: {encoder_type}.\n')

        self.encoder = encoder_hub[encoder_type](in_channels, encoder_channels, encoder_type, act_type)

    def forward(self, x):
        x3, x4, x5 = self.encoder(x)

        return x3, x4, x5


class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        hid_channels = int(in_channels // 4)
        self.act_type = act_type

        self.pool1 = self._make_pool_layer(in_channels, hid_channels, 1)
        self.pool2 = self._make_pool_layer(in_channels, hid_channels, 2)
        self.pool3 = self._make_pool_layer(in_channels, hid_channels, 4)
        self.conv = conv3x3(hid_channels, out_channels)

    def _make_pool_layer(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
                        nn.AdaptiveAvgPool2d(pool_size),
                        ConvBNAct(in_channels, out_channels, 1, act_type=self.act_type)
                )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.pool3(x), size, mode='bilinear', align_corners=True)
        x = self.conv(x1 + x2 + x3)

        return x


class STDCBackbone(nn.Module):
    def __init__(self, in_channels, encoder_channels, encoder_type, act_type):
        super().__init__()
        repeat_times_hub = {'stdc1': [1,1,1], 'stdc2': [3,4,2]}
        repeat_times = repeat_times_hub[encoder_type]
        self.stage1 = ConvBNAct(in_channels, encoder_channels[0], 3, 2)
        self.stage2 = ConvBNAct(encoder_channels[0], encoder_channels[1], 3, 2)
        self.stage3 = self._make_stage(encoder_channels[1], encoder_channels[2], repeat_times[0], act_type)
        self.stage4 = self._make_stage(encoder_channels[2], encoder_channels[3], repeat_times[1], act_type)
        self.stage5 = self._make_stage(encoder_channels[3], encoder_channels[4], repeat_times[2], act_type)

    def _make_stage(self, in_channels, out_channels, repeat_times, act_type):
        layers = [STDCModule(in_channels, out_channels, 2, act_type)]

        for _ in range(repeat_times):
            layers.append(STDCModule(out_channels, out_channels, 1, act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x3, x4, x5


class STDCModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super().__init__()
        if out_channels % 8 != 0:
            raise ValueError('Output channel should be evenly divided by 8.\n')
        self.stride = stride
        self.block1 = ConvBNAct(in_channels, out_channels//2, 1)
        self.block2 = ConvBNAct(out_channels//2, out_channels//4, 3, stride)
        if self.stride == 2:
            self.pool = nn.AvgPool2d(3, 2, 1)
        self.block3 = ConvBNAct(out_channels//4, out_channels//8, 3)
        self.block4 = ConvBNAct(out_channels//8, out_channels//8, 3)

    def forward(self, x):
        x = self.block1(x)
        x2 = self.block2(x)
        if self.stride == 2:
            x = self.pool(x)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return torch.cat([x, x2, x3, x4], dim=1)
