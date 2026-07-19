"""
Prithvi-EO-1.0-100M encoder loaded from Hugging Face with the bands and stats it consumes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from huggingface_hub import hf_hub_download

_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
_WEIGHTS_FILE = "Prithvi_EO_V1_100M.pt"
_MODEL_FILE = "prithvi_mae.py"
_CONFIG_FILE = "config.json"

# Sentinel-2 L2A earth-search asset keys spectrally matching Prithvi's HLS B02-B07 bands
BAND_NAMES: tuple[str, ...] = ("blue", "green", "red", "nir08", "swir16", "swir22")
TILE_SIZE = 224

# Prithvi's per-band mean and std are reflectance times 10000, so divide to reflectance space
_DN_SCALE = 10000.0
NORMALIZE_MEAN: tuple[float, ...] = tuple(
    v / _DN_SCALE
    for v in (
        775.2290211032589,
        1080.992780391705,
        1228.5855250417867,
        2497.2022620507532,
        2204.2139147975554,
        1610.8324823273745,
    )
)
NORMALIZE_STD: tuple[float, ...] = tuple(
    v / _DN_SCALE
    for v in (
        1281.526139861424,
        1270.0297974547493,
        1399.4802505642526,
        1368.3446143747644,
        1291.6764008585435,
        1154.505683480695,
    )
)


class PrithviInference:
    def __init__(self, encoder: torch.nn.Module) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = encoder.eval().to(self._device)

    def __call__(self, tiles: np.ndarray) -> np.ndarray:
        """Run the encoder forward over a tile batch and return pooled embeddings.

        Args:
            tiles: (n_tiles, bands, tile, tile) standardized float tile batch.

        Returns:
            (n_tiles, embed_dim) embedding array, mean-pooled over patch tokens.
        """
        with torch.no_grad():
            batch = torch.from_numpy(np.ascontiguousarray(tiles)).float().to(self._device)
            # PrithviViT takes (batch, channels, time, height, width), one timestep here.
            features = self._encoder.forward_features(batch.unsqueeze(2))
            tokens = features[-1][:, 1:]
        return tokens.mean(dim=1).cpu().numpy()


def _load_prithvi_module(path: str) -> ModuleType:
    # Import the repo's standalone prithvi_mae.py from its downloaded file path
    spec = importlib.util.spec_from_file_location("prithvi_mae", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoder_weights(state: dict) -> dict:
    # Keep the encoder tensors, dropping the decoder, the mask token, and the frame-dependent
    # positional embedding that the single-timestep encoder rebuilds at init.
    weights = state.get("model", state)
    return {
        key: value
        for key, value in weights.items()
        if not key.startswith("decoder") and key != "mask_token" and "pos_embed" not in key
    }


def build() -> PrithviInference:
    """Build the Prithvi encoder from Hugging Face weights for single-timestep inference.

    Returns:
        A callable mapping a standardized tile-batch ndarray to an embedding ndarray.
    """
    model_path = hf_hub_download(_REPO_ID, _MODEL_FILE)
    config_path = hf_hub_download(_REPO_ID, _CONFIG_FILE)
    weights_path = hf_hub_download(_REPO_ID, _WEIGHTS_FILE)
    prithvi = _load_prithvi_module(model_path)
    cfg = json.loads(Path(config_path).read_text())["pretrained_cfg"]
    encoder = prithvi.PrithviViT(
        img_size=cfg["img_size"],
        patch_size=tuple(cfg["patch_size"]),
        num_frames=1,
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=float(cfg["mlp_ratio"]),
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(_encoder_weights(state), strict=False)
    return PrithviInference(encoder)
