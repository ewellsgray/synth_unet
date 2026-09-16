"""
Configurable U-Net for multi-class semantic segmentation.

Outputs raw logits of shape (B, num_classes, H, W). Softmax over the class
dimension turns each channel into one of the 7 grayscale probability maps
(epidermis, dermis, BCC, SCC, adipose, appendageal, background) — softmax is
applied in the loss function / inference code, not inside the model, which is
the standard convention for numerical stability with CrossEntropyLoss.
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2 — the basic U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downscaling: maxpool then double conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling then double conv, with a skip connection from the encoder."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd input sizes (patch dims not perfectly divisible by 2**depth)
        # by center-cropping/padding the upsampled tensor to match the skip tensor.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(
            x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TissueSegUNet(nn.Module):
    """
    Standard encoder-decoder U-Net with skip connections.

    Args:
        in_channels: input image channels (3 for the RGB-like digitally-stained image).
        num_classes: number of output classes (7 tissue/margin categories).
        base_features: channel count after the first conv block (doubles each
            downsampling stage — this is the main knob for model capacity /
            GPU memory usage).
        depth: number of down/up-sampling stages.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        base_features: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth

        self.in_conv = DoubleConv(in_channels, base_features)

        # Build encoder (downsampling) stages, doubling channels each time.
        self.downs = nn.ModuleList()
        ch = base_features
        enc_channels = [ch]
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2
            enc_channels.append(ch)

        # Build decoder (upsampling) stages, mirroring the encoder with skips.
        self.ups = nn.ModuleList()
        for i in range(depth):
            skip_ch = enc_channels[depth - 1 - i]
            self.ups.append(Up(ch, skip_ch, skip_ch))
            ch = skip_ch

        self.out_conv = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.in_conv(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)

        return self.out_conv(x)  # (B, num_classes, H, W) raw logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape sanity check — run with `python -m model.unet`
    m = TissueSegUNet(in_channels=3, num_classes=7, base_features=32, depth=4)
    dummy = torch.randn(2, 3, 256, 256)
    out = m(dummy)
    print(f"Output shape: {tuple(out.shape)}  (expect (2, 7, 256, 256))")
    print(f"Trainable parameters: {count_parameters(m):,}")
