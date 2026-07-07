import random
import numpy as np


def expand_bbox_xyxy(bbox, img_w, img_h, margin=0.12):
    x1, y1, x2, y2 = bbox
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    return [
        max(0, x1 - mx),
        max(0, y1 - my),
        min(img_w, x2 + mx),
        min(img_h, y2 + my)
    ]


def jitter_bbox_xyxy(bbox, img_w, img_h, shift_ratio=0.05, scale_ratio=0.05):
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dx = random.uniform(-shift_ratio, shift_ratio) * bw
    dy = random.uniform(-shift_ratio, shift_ratio) * bh
    sx = 1.0 + random.uniform(-scale_ratio, scale_ratio)
    sy = 1.0 + random.uniform(-scale_ratio, scale_ratio)

    new_bw = bw * sx
    new_bh = bh * sy
    new_cx = cx + dx
    new_cy = cy + dy

    x1n = int(round(new_cx - new_bw / 2))
    y1n = int(round(new_cy - new_bh / 2))
    x2n = int(round(new_cx + new_bw / 2))
    y2n = int(round(new_cy + new_bh / 2))

    x1n = max(0, min(img_w - 1, x1n))
    y1n = max(0, min(img_h - 1, y1n))
    x2n = max(x1n + 1, min(img_w, x2n))
    y2n = max(y1n + 1, min(img_h, y2n))

    return [x1n, y1n, x2n, y2n]


def clip_bbox_xyxy(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))
    return [x1, y1, x2, y2]


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def crop_rgb_and_xyz(rgb, xyz, eye_bbox, margin=0.12):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = expand_bbox_xyxy(eye_bbox, w, h, margin)
    rgb_crop = rgb[y1:y2, x1:x2]
    xyz_crop = xyz[y1:y2, x1:x2] if xyz is not None else None
    return rgb_crop, xyz_crop, (x1, y1)


def bbox_global_to_local(bbox, offset):
    ox, oy = offset
    x1, y1, x2, y2 = bbox
    return [x1 - ox, y1 - oy, x2 - ox, y2 - oy]


def bbox_local_to_global(bbox, offset):
    ox, oy = offset
    x1, y1, x2, y2 = bbox
    return [x1 + ox, y1 + oy, x2 + ox, y2 + oy]


def xyxy_to_yolo(bbox, img_w, img_h, class_id=0):
    x1, y1, x2, y2 = bbox
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    return [
        class_id,
        xc / img_w,
        yc / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    ]


def yolo_to_xyxy(label, img_w, img_h):
    _, xc, yc, wn, hn = label
    w = wn * img_w
    h = hn * img_h
    return [
        int(xc * img_w - w / 2),
        int(yc * img_h - h / 2),
        int(xc * img_w + w / 2),
        int(yc * img_h + h / 2),
    ]


def get_3d_from_bbox(xyz_crop, bbox_local):
    x1, y1, x2, y2 = [int(v) for v in bbox_local]

    # Inner margin to avoid specular edges from the metallic trocar
    margin_px = 2
    roi = xyz_crop[
        y1 + margin_px: y2 - margin_px,
        x1 + margin_px: x2 - margin_px
    ].reshape(-1, 3)

    # Remove NaN, inf and zero points
    finite = np.isfinite(roi).all(axis=1)
    valid = roi[finite]
    if len(valid) == 0:
        return None

    nonzero = np.linalg.norm(valid, axis=1) > 1e-6
    valid = valid[nonzero]
    if len(valid) == 0:
        return None

    # IQR filtering per axis
    for i in range(3):
        col = valid[:, i]
        p25, p75 = np.percentile(col, [25, 75])
        iqr = p75 - p25
        mask = (col >= p25 - 1.5 * iqr) & (col <= p75 + 1.5 * iqr)
        valid = valid[mask]

    if len(valid) == 0:
        return None

    # Remove points far from a provisional median
    median = np.median(valid, axis=0)
    dists = np.linalg.norm(valid - median, axis=1)
    threshold = np.percentile(dists, 75)
    valid = valid[dists <= threshold]

    if len(valid) == 0:
        return None

    if len(valid) < 5:
        print(f"[WARN] Only {len(valid)} valid points after filtering")

    return np.median(valid, axis=0)


def get_3d_from_mask(xyz_crop, seg_mask):
    """
    xyz_crop : np.ndarray (H, W, 3) float32
        Aligned XYZ crop.

    seg_mask : np.ndarray (H, W) uint8/bool
        Binary trocar mask.

    Returns:
        (centroid_3d, trocar_pointcloud) or (None, None)
    """
    points = xyz_crop[seg_mask > 0]   # (N, 3) — only trocar pixels

    # Same robust filters used in get_3d_from_bbox
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        return None, None

    nonzero = np.linalg.norm(points, axis=1) > 1e-6
    points = points[nonzero]
    if len(points) == 0:
        return None, None

    for i in range(3):
        col = points[:, i]
        p25, p75 = np.percentile(col, [25, 75])
        iqr = p75 - p25
        mask = (col >= p25 - 1.5 * iqr) & (col <= p75 + 1.5 * iqr)
        points = points[mask]

    if len(points) == 0:
        return None, None

    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)
    points = points[dists <= np.percentile(dists, 75)]

    if len(points) == 0:
        return None, None

    if len(points) < 5:
        print(f"[WARN] Only {len(points)} valid points in the trocar mask")

    centroid = np.median(points, axis=0)   # robust 3D center

    return centroid, points                # points = segmented trocar point cloud