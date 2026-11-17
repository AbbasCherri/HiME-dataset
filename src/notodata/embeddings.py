"""D3 embedding cache: one frozen-encoder vector per `ok` chip, behind a sidecar.

Files per (encoder, layer): `<layer>.npy` (float32, one row per `ok` chip) and
`<layer>.ids.npy` (the matching `building_id`s, ascending). A chip that is not
`ok` has no row, and `lookup` returns `None` for it, never a zero vector.

The sidecar adds `chip_index_sha256` (a hash of the chip pixel hashes) and
`deterministic` to the D3 keys, so changed chip pixels invalidate embeddings
even when `PIPELINE_VERSION` did not move.

`MeanColourEncoder` and `DinoV2Encoder` are built. The other candidate encoders
are registry entries without weights; building them is the optional Task 27.
An encoder may carry `sidecar_meta` (a dict), merged into the sidecar so that its
model and preprocessing take part in the cache key. An encoder may also carry
`provenance` (library versions, known once the model has run); `ensure_embeddings`
writes it to `<layer>.provenance.json`, outside the cache key.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from PIL import Image

from notodata import cache, chips, paths
from notodata.config import CONFIG
from notodata.io import atomic_write, sha256_bytes

log = logging.getLogger(__name__)

BATCH_SIZE = 64
"""Fixed so CPU builds do not depend on how many chips happen to be left."""


class Encoder(Protocol):
    """Resizing and normalisation live inside `embed`, so the id fixes the output."""

    encoder_id: str
    """Name, weights hash and preprocessing in one string."""
    dim: int
    deterministic: bool
    """False for GPU builds; recorded in the sidecar."""

    def embed(self, batch_uint8: np.ndarray) -> np.ndarray:
        """`uint8[B, H, W, 3]` to `float32[B, dim]`."""
        ...


class MeanColourEncoder:
    """Per-channel mean in [0, 1]. Tiny and exact, for offline tests."""

    encoder_id = "mean_colour"
    dim = 3
    deterministic = True

    def embed(self, batch_uint8: np.ndarray) -> np.ndarray:
        mean = batch_uint8.astype(np.float64).mean(axis=(1, 2)) / 255.0
        return mean.astype(np.float32)


def dinov2_preprocessing_note() -> str:
    cfg = CONFIG.dinov2
    return (
        f"timm img_size={cfg.input_px}, "
        f"resize {cfg.input_px}x{cfg.input_px} {cfg.interpolation} (PIL, uint8), /255, "
        f"mean {list(cfg.mean)}, std {list(cfg.std)}, NCHW; no L2 normalisation"
    )


def dinov2_preprocess(batch_uint8: np.ndarray) -> np.ndarray:
    """`uint8[B, H, W, 3]` to `float32[B, 3, px, px]`, ImageNet-normalised."""
    cfg = CONFIG.dinov2
    resample = getattr(Image.Resampling, cfg.interpolation.upper())
    size = (cfg.input_px, cfg.input_px)
    resized = np.stack([np.asarray(Image.fromarray(a).resize(size, resample)) for a in batch_uint8])
    scaled = resized.astype(np.float32) / 255.0
    mean = np.asarray(cfg.mean, dtype=np.float32)
    std = np.asarray(cfg.std, dtype=np.float32)
    return np.ascontiguousarray(((scaled - mean) / std).transpose(0, 3, 1, 2))


def dinov2_create_kwargs() -> dict[str, Any]:
    """`img_size` makes timm resample the position embedding; the default 518 would reject 224."""
    return {"pretrained": True, "num_classes": 0, "img_size": CONFIG.dinov2.input_px}


def _timm_dinov2_forward() -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, str]]:
    """Load the pretrained timm model; torch and timm are imported only here."""
    import timm
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(CONFIG.dinov2.timm_name, **dinov2_create_kwargs())
    model = model.eval().to(device)

    def forward(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(x).to(device)).float().cpu().numpy()

    return forward, {"timm": timm.__version__, "torch": torch.__version__}


class DinoV2Encoder:
    """DINOv2 ViT-S/14 pooled features through timm; weights load on first `embed`."""

    encoder_id = "dinov2_vits14"
    dim = 384
    deterministic = False

    def __init__(self, forward: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        """`forward` maps `float32[B, 3, px, px]` to `[B, dim]`; tests inject a fake."""
        self._forward = forward
        self.provenance: dict[str, str] = {}
        """Library versions, set when the real model loads; kept out of the cache key."""
        self.sidecar_meta: dict[str, Any] = {
            "encoder_model": CONFIG.dinov2.timm_name,
            "encoder_preprocessing": dinov2_preprocessing_note(),
        }

    def embed(self, batch_uint8: np.ndarray) -> np.ndarray:
        if self._forward is None:
            self._forward, self.provenance = _timm_dinov2_forward()
        out = np.asarray(self._forward(dinov2_preprocess(batch_uint8)), dtype=np.float32)
        if out.shape != (len(batch_uint8), self.dim):
            raise ValueError(f"expected features of width {self.dim}, got {out.shape}")
        return out


@dataclass(frozen=True)
class EncoderSpec:
    encoder_id: str
    dim: int
    licence_note: str


REGISTRY: dict[str, EncoderSpec] = {
    spec.encoder_id: spec
    for spec in (
        EncoderSpec("mean_colour", 3, "test encoder, no weights"),
        EncoderSpec(
            "resnet50.tv_in1k_v2",
            2048,
            "torchvision/timm weights; licence to be verified in Task 27",
        ),
        EncoderSpec(
            "dinov2_vits14",
            384,
            "DINOv2 weights are Apache-2.0 via timm/HF; to be re-checked at release",
        ),
        EncoderSpec(
            "openclip_vitb32_laion2b",
            512,
            "OpenCLIP LAION-2B weights; licence to be verified in Task 27",
        ),
        EncoderSpec(
            "satlas_aerial_swinb_si",
            0,
            "Satlas aerial Swin-B weights; licence and output dim to be verified in Task 27",
        ),
    )
}
"""`dim` 0 means not yet pinned."""


def get_encoder(encoder_id: str) -> Encoder:
    if encoder_id not in REGISTRY:
        raise KeyError(f"unknown encoder {encoder_id!r}; registered: {sorted(REGISTRY)}")
    if encoder_id == MeanColourEncoder.encoder_id:
        return MeanColourEncoder()
    if encoder_id == DinoV2Encoder.encoder_id:
        return DinoV2Encoder()
    raise NotImplementedError(f"{encoder_id}: real encoders are built in Task 27 (needs [embed])")


# -- storage ---------------------------------------------------------------


def embedding_path(encoder_id: str, layer_id: str) -> Path:
    return paths.embeddings_dir(encoder_id) / f"{layer_id}.npy"


def ids_path(encoder_id: str, layer_id: str) -> Path:
    return paths.embeddings_dir(encoder_id) / f"{layer_id}.ids.npy"


def provenance_path(encoder_id: str, layer_id: str) -> Path:
    return paths.embeddings_dir(encoder_id) / f"{layer_id}.provenance.json"


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def _ok_rows(index: pd.DataFrame) -> pd.DataFrame:
    ok = index[index["status"] == "ok"]
    return ok.sort_values("building_id", kind="mergesort").reset_index(drop=True)


def chip_index_sha256(index: pd.DataFrame) -> str:
    """Hash of the `ok` chips' pixel hashes, in building-id order."""
    ok = _ok_rows(index)
    pairs = [[int(b), s] for b, s in zip(ok["building_id"], ok["sha256"], strict=True)]
    return sha256_bytes(json.dumps(pairs, separators=(",", ":")).encode())


def _load_chip(row) -> np.ndarray:
    with Image.open(paths.notodata_home() / row.path) as img:
        pixels = np.asarray(img.convert("RGB"))
    if chips.pixel_sha256(pixels) != row.sha256:
        raise ValueError(
            f"chip {row.building_id} pixels do not match the chip index; rebuild the chips"
        )
    return pixels


def ensure_embeddings(
    encoder: Encoder, layer_id: str, index: pd.DataFrame, *, batch_size: int = BATCH_SIZE
) -> bool:
    """Keep the embedding file for (encoder, layer) valid for `index`; True if rebuilt."""
    ok = _ok_rows(index)
    path = embedding_path(encoder.encoder_id, layer_id)
    ids_file = ids_path(encoder.encoder_id, layer_id)
    meta = cache.expected_meta(
        layer_ids=[layer_id],
        zoom=CONFIG.tile_chip.zoom,
        chip_size=CONFIG.tile_chip.chip_size_px,
        encoder_id=encoder.encoder_id,
        building_ids=ok["building_id"],
    )
    meta["chip_index_sha256"] = chip_index_sha256(index)
    meta["deterministic"] = bool(encoder.deterministic)
    meta.update(getattr(encoder, "sidecar_meta", {}))

    if not ids_file.exists():
        # the sidecar vouches for both files; without the ids it vouches for nothing
        cache.sidecar_path(path).unlink(missing_ok=True)

    def compute(p: Path) -> None:
        rows = list(ok.itertuples(index=False))
        out = np.empty((len(rows), encoder.dim), dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = np.stack([_load_chip(r) for r in chunk])
            out[start : start + len(chunk)] = encoder.embed(batch)
        ids = ok["building_id"].to_numpy(dtype=np.int64)
        atomic_write(ids_file, _npy_bytes(ids))
        atomic_write(p, _npy_bytes(out))
        versions = getattr(encoder, "provenance", {})
        if versions:
            blob = json.dumps(versions, indent=2, sort_keys=True) + "\n"
            atomic_write(provenance_path(encoder.encoder_id, layer_id), blob.encode())

    return cache.ensure(path, meta, compute)


@dataclass(frozen=True)
class EmbeddingStore:
    ids: np.ndarray
    vectors: np.ndarray

    def get(self, building_id: int) -> np.ndarray | None:
        pos = int(np.searchsorted(self.ids, building_id))
        if pos < len(self.ids) and self.ids[pos] == building_id:
            return self.vectors[pos]
        return None

    def lookup(self, building_ids) -> list[np.ndarray | None]:
        return [self.get(int(b)) for b in building_ids]


def load(encoder_id: str, layer_id: str) -> EmbeddingStore:
    ids = np.load(ids_path(encoder_id, layer_id), allow_pickle=False)
    vectors = np.load(embedding_path(encoder_id, layer_id), allow_pickle=False)
    if len(ids) != len(vectors):
        raise ValueError(f"{encoder_id}/{layer_id}: {len(ids)} ids but {len(vectors)} rows")
    return EmbeddingStore(ids=ids, vectors=vectors)
