"""
A small ViT defined in torch together with the bands and tile size it consumes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

BAND_NAMES: tuple[str, ...] = ("red", "green", "blue", "nir")
TILE_SIZE = 224


class _ViT(nn.Module):
    # Patchify, run a transformer encoder, return the class-token embedding per tile
    def __init__(
        self,
        in_channels: int,
        tile_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        n_patches = (tile_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(n, -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.pos_embed
        encoded = self.norm(self.encoder(tokens))
        return encoded[:, 0]


class ViTInference:
    def __init__(self, module: nn.Module) -> None:
        self._module = module.eval()

    def __call__(self, tiles: np.ndarray) -> np.ndarray:
        """Run the forward pass over a tile batch and return the embeddings.

        Args:
            tiles: (n_tiles, bands, tile, tile) float tile batch.

        Returns:
            (n_tiles, embed_dim) embedding array.
        """
        with torch.no_grad():
            batch = torch.from_numpy(np.ascontiguousarray(tiles)).float()
            out = self._module(batch)
        return out.numpy()


def build() -> ViTInference:
    """Build the small ViT with one input channel per declared band.

    Returns:
        A callable mapping a tile-batch ndarray to an embedding ndarray.
    """
    module = _ViT(
        in_channels=len(BAND_NAMES),
        tile_size=TILE_SIZE,
        patch_size=16,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4.0,
    )
    return ViTInference(module)
