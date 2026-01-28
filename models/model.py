"""
IRNO Model Components.

RefinementOperator (Φ_θ): U-Net architecture that learns residual corrections.
"""

import torch
import torch.nn as nn


class RefinementOperator(nn.Module):
    """
    U-Net based refinement operator Φ_θ for IRNO.

    Takes concatenated (x, h_k) as input and outputs residual correction.
    Uses circular padding for periodic boundary conditions.

    Args:
        in_channels: Number of input channels (input fields + current estimate)
        out_channels: Number of output channels (prediction fields)
        base_channels: Base channel count (doubles at each level)
        depth: Number of encoder/decoder levels
        padding_type: 'circular' for periodic domains, None for zero padding
    """

    def __init__(self, in_channels, out_channels, base_channels=16, depth=4, padding_type="circular"):
        super().__init__()

        self.depth = depth
        self.padding_type = padding_type

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = base_channels
        for i in range(depth):
            in_ch = in_channels if i == 0 else ch // 2
            self.encoders.append(self._conv_block(in_ch, ch))
            self.pools.append(nn.MaxPool2d(2))
            ch *= 2

        # Bottleneck
        self.bottleneck = self._conv_block(ch // 2, ch)

        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth):
            self.upconvs.append(self._up_block(ch, ch // 2))
            self.decoders.append(self._conv_block(ch, ch // 2))
            ch //= 2

        # Output layer with small initialization for stable refinement
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)
        nn.init.zeros_(self.output.bias)

    def _conv_block(self, in_ch, out_ch):
        """Double convolution block with BatchNorm and GELU."""
        padding_mode = self.padding_type if self.padding_type else "zeros"
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def _up_block(self, in_ch, out_ch):
        """Upsample + convolution block."""
        padding_mode = self.padding_type if self.padding_type else "zeros"
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode)
        )

    def forward(self, x):
        """
        Args:
            x: Concatenated input [x, h_k] with shape (B, C_in, H, W)

        Returns:
            Residual correction with shape (B, C_out, H, W)
        """
        skip_connections = []

        # Encoder
        for i in range(self.depth):
            x = self.encoders[i](x)
            skip_connections.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.decoders[i](x)

        return self.output(x)


def load_base_operator(pretrained_name, device):
    """
    Load pretrained FNO as base operator T_base from HuggingFace.

    Args:
        pretrained_name: HuggingFace model name (e.g., "polymathic-ai/FNO-active_matter")
        device: Device to load model on

    Returns:
        Frozen FNO model in eval mode
    """
    from the_well.benchmark.models import FNO

    model = FNO.from_pretrained(pretrained_name)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model
