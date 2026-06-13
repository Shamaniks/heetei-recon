"""
Stage 1 – Raw point ingestion from a .laz file.

PATCH v1.1 — Raw integer ingestion path
----------------------------------------
The original code called np.array(las.x, dtype=np.float32), which forces
laspy to first materialise a full float64 ScaledArrayView (26.5M × 8 B =
212 MB) before NumPy can downcast it to float32. With three coordinates that
is 636 MB of invisible float64 intermediates on the C heap — enough to tip
Colab into OOM before a single Open3D object is created.

Fix: read the raw int32 storage arrays (las.X / las.Y / las.Z), cast them
directly to float32, then apply the LAS scale and offset using float32 scalar
arithmetic. The float64 intermediate is never materialised.

Memory budget (26.5M points):
  int32 per axis   →  26.5M × 4 B = 106 MB  (temporary, freed after cast)
  float32 per axis →  26.5M × 4 B = 106 MB  (kept)
  float32 xyz      →  26.5M × 12 B = 318 MB (column-stacked result)
  Peak delta vs original: saves ~318 MB of hidden float64 allocations.
"""

import laspy
import numpy as np


def load_point_cloud(laz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Open a .laz file and return (xyz, intensity).

    Returns
    -------
    xyz       : np.ndarray, shape (N, 3), dtype float32
    intensity : np.ndarray, shape (N,),   dtype uint8
    """
    las = laspy.read(laz_path)

    # ── PATCH: raw integer → float32 conversion ───────────────────────────
    # las.X / las.Y / las.Z are the raw int32 storage values (no scaling).
    # las.header.scale and las.header.offset are per-axis float64 scalars —
    # we cast them to float32 so the arithmetic never promotes the array.
    #
    # Equivalent to the LAS spec formula:
    #   coordinate = (raw_int * scale) + offset
    # but executed entirely in float32 space.

    scale  = np.float32(las.header.scale)    # shape (3,) after float32 cast
    offset = np.float32(las.header.offset)   # shape (3,)

    # np.array(las.X, dtype=np.float32) reads the raw int32 store and
    # upcasts directly to float32 — no float64 intermediate is created.
    x_raw = np.array(las.X, dtype=np.float32)
    x = x_raw * scale[0] + offset[0]
    del x_raw   # free the intermediate int32 cast immediately

    y_raw = np.array(las.Y, dtype=np.float32)
    y = y_raw * scale[1] + offset[1]
    del y_raw

    z_raw = np.array(las.Z, dtype=np.float32)
    z = z_raw * scale[2] + offset[2]
    del z_raw
    # ── end PATCH ─────────────────────────────────────────────────────────

    xyz = np.column_stack((x, y, z))
    del x, y, z   # column_stack copies; originals are now redundant

    assert xyz.ndim == 2 and xyz.shape[1] == 3, (
        f"Expected (N, 3) point array, got shape {xyz.shape}"
    )

    # Intensity: uint16 in LAS PF1, but EDA confirms values ≤ 255.
    # Cast to uint8 immediately — saves 26.5M bytes vs keeping uint16.
    intensity = np.array(las.intensity, dtype=np.uint8)
    assert intensity.shape[0] == xyz.shape[0], (
        "Intensity length does not match point count — file may be corrupt"
    )

    print(f"[ingest] Loaded {xyz.shape[0]:,} points from '{laz_path}'")
    return xyz, intensity
