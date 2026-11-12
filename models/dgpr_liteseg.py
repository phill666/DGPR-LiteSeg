"""Frozen integration with publication-facing names; operation/state order retained.

Source and distribution terms: see LICENSE and README.md.
"""
import torch.nn as nn
import torch.nn.functional as F
from .base_ops import conv1x1, ConvBNAct
from .backbone import Encoder, SPPM
from .dgff import DGFF
from .gpcr import GPCR


class DGFFFusion(nn.Module):
    def __init__(self, low_channels, out_channels):
        super().__init__()
        self.low_projection = conv1x1(low_channels, out_channels)
        self.fusion = DGFF(out_channels)

    def forward(self, x_high, x_low):
        x_low = self.low_projection(x_low)
        x_high = F.interpolate(x_high, size=x_low.shape[-2:],
                               mode='bilinear', align_corners=True)
        return self.fusion(x_high, x_low)


class Decoder(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stage6 = ConvBNAct(32, 32)
        self.fusion1 = DGFFFusion(512, 32)
        self.refine1 = GPCR(32)
        self.stage7 = ConvBNAct(32, 64)
        self.fusion2 = DGFFFusion(256, 64)
        # These frozen Identity placeholders preserve the historical object graph.
        self.edgeflow2 = nn.Identity()
        self.refine2 = GPCR(64)
        self.uprc2 = nn.Identity()
        self.stage8 = ConvBNAct(64, 128)
        self.seg_head = ConvBNAct(128, num_classes, 3, act_type='relu')

    def forward(self, x3, x4, x5, output_size):
        x = self.stage6(x5)
        x = self.refine1(self.fusion1(x, x4))
        x = self.stage7(x)
        x = self.fusion2(x, x3)
        x = self.edgeflow2(x)
        x = self.refine2(x)
        x = self.uprc2(x)
        x = self.stage8(x)
        x = self.seg_head(x)
        return F.interpolate(x, output_size, mode='bilinear', align_corners=True)


class DGPRLiteSeg(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        channels = [32, 64, 256, 512, 1024]
        self.encoder = Encoder(3, channels, 'stdc1', 'relu')
        self.sppm = SPPM(1024, 32, 'relu')
        self.use_sapa = False
        self.sapa = None
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x3, x4, x5 = self.encoder(x)
        return self.decoder(x3, x4, self.sppm(x5), x.shape[-2:])


def build_model(num_classes=2):
    """Construct only the frozen DGPR model; no checkpoint is required."""
    return DGPRLiteSeg(num_classes=num_classes)
