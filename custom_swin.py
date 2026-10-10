"""KidneySwin: a from-scratch Swin-style 2D image classifier.

The entire architecture is implemented here using only PyTorch primitives.
No timm, torchvision models, or pretrained weights are used.

This is a research prototype, not a medical diagnostic device.
"""

import torch
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth for [B,H,W,C] tensors."""
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert [B,H,W,C] to [B*num_windows, window_size**2, C]."""
    batch, height, width, channels = x.shape
    if height % window_size or width % window_size:
        raise ValueError("Feature dimensions must be divisible by window_size")
    x = x.reshape(
        batch, height // window_size, window_size,
        width // window_size, window_size, channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
        -1, window_size * window_size, channels
    )


def window_reverse(windows: torch.Tensor, window_size: int,
                   height: int, width: int, batch: int) -> torch.Tensor:
    """Convert windows back to [B,H,W,C]."""
    channels = windows.shape[-1]
    x = windows.reshape(
        batch, height // window_size, width // window_size,
        window_size, window_size, channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(
        batch, height, width, channels
    )


class WindowSelfAttention(nn.Module):
    """Local multi-head attention with learned relative position bias."""

    def __init__(self, dim: int, num_heads: int, window_size: int, attn_drop: float = 0.05):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(attn_drop)

        window_tokens = window_size ** 2
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        coordinates = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        )).flatten(1)  # [2, window_tokens]
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        index = relative.sum(dim=-1)
        self.register_buffer("relative_position_index", index, persistent=False)
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

    def forward(self, windows: torch.Tensor,
                attention_mask: torch.Tensor | None,
                num_windows: int) -> torch.Tensor:
        batch_windows, tokens, channels = windows.shape
        qkv = self.qkv(windows).reshape(
            batch_windows, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.num_heads).permute(2, 0, 1)
        scores = scores + bias.unsqueeze(0)

        if attention_mask is not None:
            batch = batch_windows // num_windows
            scores = scores.reshape(batch, num_windows, self.num_heads, tokens, tokens)
            scores = scores + attention_mask.unsqueeze(0).unsqueeze(2)
            scores = scores.reshape(batch_windows, self.num_heads, tokens, tokens)

        probabilities = self.attn_drop(scores.softmax(dim=-1))
        attended = (probabilities @ v).transpose(1, 2).reshape(
            batch_windows, tokens, channels
        )
        return self.proj_drop(self.proj(attended))


def build_shift_attention_mask(height: int, width: int,
                               window_size: int, shift_size: int) -> torch.Tensor:
    """Prevent cyclic wrapping from creating invalid attention connections."""
    labels = torch.zeros((1, height, width, 1))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    label = 0
    for h in h_slices:
        for w in w_slices:
            labels[:, h, w, :] = label
            label += 1
    labels = window_partition(labels, window_size).squeeze(-1)
    differences = labels.unsqueeze(1) - labels.unsqueeze(2)
    return differences.masked_fill(differences != 0, -100.0).masked_fill(
        differences == 0, 0.0
    )


class SwinBlock(nn.Module):
    """Alternating unshifted and shifted-window Swin block."""

    def __init__(self, dim: int, resolution: tuple[int, int],
                 num_heads: int, window_size: int, shifted: bool,
                 drop_path: float = 0.0, mlp_drop: float = 0.10):
        super().__init__()
        self.height, self.width = resolution
        self.window_size = window_size
        self.shift_size = window_size // 2 if shifted else 0
        if min(self.height, self.width) <= window_size:
            self.shift_size = 0
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowSelfAttention(dim, num_heads, window_size)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(mlp_drop),
            nn.Linear(4 * dim, dim), nn.Dropout(mlp_drop)
        )
        if self.shift_size:
            mask = build_shift_attention_mask(
                self.height, self.width, window_size, self.shift_size
            )
        else:
            mask = None
        self.register_buffer("attention_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        if (height, width) != (self.height, self.width):
            raise ValueError("Input resolution does not match configured block")
        shortcut = x
        x = self.norm1(x)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size)
        num_windows = (height // self.window_size) * (width // self.window_size)
        windows = self.attention(windows, self.attention_mask, num_windows)
        x = window_reverse(windows, self.window_size, height, width, batch)
        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """Downsample spatially 2x, while doubling the feature width."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, height, width, _ = x.shape
        if height % 2 or width % 2:
            raise ValueError("Patch merging needs even spatial dimensions")
        x = torch.cat(
            (x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
             x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]), dim=-1,
        )
        return self.reduction(self.norm(x))


class KidneySwin(nn.Module):
    """User-defined 4-stage Swin Transformer trained with random initialization.

    Defaults deliberately differ from the standard Swin-Tiny configuration:
    48 initial channels, stage depths [2,2,4,2], 1-channel CT input.
    """

    def __init__(self, image_size: int = 224, in_chans: int = 1,
                 num_classes: int = 4, embed_dim: int = 48,
                 depths: tuple[int, ...] = (2, 2, 4, 2),
                 num_heads: tuple[int, ...] = (3, 6, 12, 24),
                 window_size: int = 7, patch_size: int = 4,
                 drop_path_rate: float = 0.08, drop_rate: float = 0.10):
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("Exactly four stages are required")
        if image_size % (patch_size * 8):
            raise ValueError("image_size must be divisible by patch_size*8")
        sizes = [image_size // patch_size // (2 ** i) for i in range(4)]
        if any(size % window_size for size in sizes):
            raise ValueError("Stage resolutions must be divisible by window_size")
        self.image_size = image_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        drop_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        for stage_number in range(4):
            channels = embed_dim * (2 ** stage_number)
            resolution = (sizes[stage_number], sizes[stage_number])
            if channels % num_heads[stage_number]:
                raise ValueError("Each stage dimension must divide into its heads")
            blocks = nn.Sequential(*[
                SwinBlock(channels, resolution, num_heads[stage_number],
                          window_size, shifted=(block_index % 2 == 1),
                          drop_path=drop_rates[sum(depths[:stage_number]) + block_index],
                          mlp_drop=drop_rate)
                for block_index in range(depths[stage_number])
            ])
            self.stages.append(blocks)
            if stage_number < 3:
                self.mergers.append(PatchMerging(channels))
        final_channels = embed_dim * 8
        self.norm = nn.LayerNorm(final_channels)
        self.head_dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(final_channels, num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected [batch, channels, {self.image_size}, {self.image_size}]")
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < 3:
                x = self.mergers[index](x)
        x = self.norm(x).mean(dim=(1, 2))
        return self.classifier(self.head_dropout(x))  # Raw logits


if __name__ == "__main__":
    model = KidneySwin()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 224, 224))
    print("Output shape:", tuple(output.shape))
    print("Parameter count:", sum(p.numel() for p in model.parameters()))
