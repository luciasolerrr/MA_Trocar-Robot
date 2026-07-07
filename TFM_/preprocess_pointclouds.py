"""
preprocess_pointclouds.py
-------------------------

STEP 2 of the pipeline:

Convert Zivid organized point clouds from .ply format to .npy arrays.
Each .ply point cloud is expected to be organized and aligned with its
corresponding RGB image. The script uses the image resolution to reshape
the flattened point cloud into:

    xyz.shape = (H, W, 3)

Expected input structure:

data/
└── full_export/
    ├── images/
    │   ├── sample_001.png
    │   └── ...
    ├── point_clouds/
    │   ├── sample_001.ply
    │   └── ...
    └── xyz/              ← created by this script

Output:

data/full_export/xyz/*.npy

Each .npy file contains:
    dtype: float32
    shape: (H, W, 3)
    channels: X, Y, Z in millimeters
"""

import cv2
import numpy as np
from plyfile import PlyData
from pathlib import Path

PLY_DIR = Path("full_export/point_clouds")
IMG_DIR = Path("full_export/images")
OUT_DIR = Path("full_export/xyz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ok = skipped = 0

for ply_path in sorted(PLY_DIR.glob("*.ply")):
    stem = ply_path.stem
    img_path = IMG_DIR / f"{stem}.png"

    if not img_path.exists():
        print(f"  [SKIP] no image for {stem}")
        skipped += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] image could not be loaded {stem}")
        skipped += 1
        continue
    h, w = img.shape[:2]

    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if pts.shape[0] != h * w:
        print(f"  [SKIP] {stem}: points={pts.shape[0]} != H*W={h*w}")
        skipped += 1
        continue

    xyz = pts.reshape(h, w, 3)
    np.save(str(OUT_DIR / f"{stem}.npy"), xyz)
    ok += 1
    print(f"  [OK]   {stem}")

print(f"\nConverted: {ok}  |  Skipped: {skipped}")
