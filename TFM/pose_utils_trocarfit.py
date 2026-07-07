import numpy as np
import cv2


DEFAULT_TROCAR_ENTRY_OFFSET_MM = 2.0
DEFAULT_TARGET_CLEARANCE_MM = 50.0


# ============================================================
# Basic point-cloud utilities
# ============================================================

def filter_points(points):
    """
    Filter NaN, inf, zero points, and basic outliers.
    """
    if points is None or len(points) == 0:
        return None

    points = points.reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    if len(points) == 0:
        return None

    nonzero = np.linalg.norm(points, axis=1) > 1e-6
    points = points[nonzero]

    if len(points) == 0:
        return None

    # IQR filtering per axis
    for i in range(3):
        if len(points) == 0:
            return None

        col = points[:, i]
        p25, p75 = np.percentile(col, [25, 75])
        iqr = p75 - p25

        if iqr < 1e-9:
            continue

        mask = (col >= p25 - 1.5 * iqr) & (col <= p75 + 1.5 * iqr)
        points = points[mask]

    if len(points) == 0:
        return None

    return points


# ============================================================
# Sphere fitting
# ============================================================

def fit_sphere_least_squares(points):
    """
    Fit a sphere:

        x^2 + y^2 + z^2 + ax + by + cz + d = 0

    Returns:
        center, radius
    """
    points = filter_points(points)

    if points is None or len(points) < 20:
        return None, None

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    A = np.column_stack([x, y, z, np.ones_like(x)])
    b = -(x**2 + y**2 + z**2)

    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None

    a, b_, c, d = sol

    center = np.array([-a / 2, -b_ / 2, -c / 2])
    radius_sq = np.sum(center**2) - d

    if radius_sq <= 0:
        return None, None

    radius = np.sqrt(radius_sq)

    if not np.isfinite(radius):
        return None, None

    return center, radius


def sphere_from_4_points(points4):
    """
    Compute an exact sphere from 4 points.

    Returns:
        center, radius
    """
    p1, p2, p3, p4 = points4

    A = 2.0 * np.array([
        p2 - p1,
        p3 - p1,
        p4 - p1,
    ])

    b = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1),
        np.dot(p4, p4) - np.dot(p1, p1),
    ])

    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None

    radius = np.linalg.norm(p1 - center)

    if not np.isfinite(radius) or radius <= 0:
        return None, None

    return center, radius


def fit_sphere_ransac(
    points,
    n_iter=1500,
    distance_threshold=0.8,
    radius_range=(13.5, 15.5),
    min_inliers=200,
):
    """
    Robust sphere fitting using RANSAC.

    Args:
        points:
            Candidate 3D point cloud of the eye.

        distance_threshold:
            Radial tolerance in mm.

        radius_range:
            Allowed radius range in mm.

        min_inliers:
            Minimum number of inliers required to accept the sphere.

    Returns:
        center, radius, inlier_points
    """
    points = filter_points(points)

    if points is None or len(points) < 20:
        return None, None, None

    n = len(points)

    best_center = None
    best_radius = None
    best_inliers = None
    best_score = -1

    # Fixed seed for deterministic sphere fitting across runs.
    # Without this, marginal point clouds produce different eye_center_mm
    # on each execution, causing centimeter-scale target variability.
    np.random.seed(42)

    for _ in range(n_iter):
        idx = np.random.choice(n, 4, replace=False)
        sample = points[idx]

        center, radius = sphere_from_4_points(sample)

        if center is None:
            continue

        if radius < radius_range[0] or radius > radius_range[1]:
            continue

        d = np.linalg.norm(points - center, axis=1)
        residuals = np.abs(d - radius)

        inliers = residuals < distance_threshold
        score = int(np.sum(inliers))

        if score > best_score:
            best_score = score
            best_center = center
            best_radius = radius
            best_inliers = inliers

    if best_inliers is None or best_score < min_inliers:
        return None, None, None

    # Final refinement using the inliers
    inlier_points = points[best_inliers]
    center_refined, radius_refined = fit_sphere_least_squares(inlier_points)

    if center_refined is None:
        return best_center, best_radius, inlier_points

    return center_refined, radius_refined, inlier_points


# ============================================================
# Geometry
# ============================================================

def orient_axis_towards_target(p0, v, p_target):
    """
    Make the axis point from p0 towards the selected target.
    """
    if np.dot(v, p_target - p0) < 0:
        v = -v
    return v


def intersect_line_sphere(p0, v, center, radius):
    """
    Intersection between:

        L(t) = p0 + t v

    and:

        ||X - center||^2 = radius^2

    Returns the sphere intersection closest to p0.

    This is intentional. In most cases p0 is outside the eye and v points
    towards the eye center, so the closest intersection is the entry surface.
    If the mask/depth estimate places p0 slightly inside the fitted sphere,
    choosing the first positive intersection would incorrectly select the far
    exit surface. The closest intersection is the correct anatomical entry
    side for drawing the short outside cylinder and for p_entry_trocar.
    """
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        return None

    v = v / norm_v

    oc = p0 - center

    a = np.dot(v, v)
    b = 2.0 * np.dot(oc, v)
    c = np.dot(oc, oc) - radius**2

    disc = b**2 - 4 * a * c

    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)

    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    p1 = p0 + t1 * v
    p2 = p0 + t2 * v

    # Entry side = nearest surface point to the estimated trocar point.
    if abs(t1) <= abs(t2):
        return p1
    return p2


# ============================================================
# Eye point extraction
# ============================================================

def extract_eye_points_for_sphere(
    xyz_crop,
    trocar_mask,
    trocar_points=None,
    inner_margin_px=8,
    ellipse_scale=0.78,
    z_window_mm=12.0,
    dilate_trocar_px=5,
):
    """
    Extract a clean eye point cloud for sphere fitting.

    Strategy:
        - remove the trocar
        - remove an area around the trocar
        - remove crop borders
        - use a central elliptical mask
        - filter by depth
    """
    if xyz_crop is None or trocar_mask is None:
        return None

    h, w = trocar_mask.shape[:2]

    # 1. Dilated trocar mask to remove contaminated border regions
    trocar_mask_u8 = (trocar_mask > 0).astype(np.uint8)

    if dilate_trocar_px > 0:
        kernel = np.ones((dilate_trocar_px, dilate_trocar_px), np.uint8)
        trocar_mask_dilated = cv2.dilate(trocar_mask_u8, kernel, iterations=1)
    else:
        trocar_mask_dilated = trocar_mask_u8

    # 2. Initial valid mask: non-trocar pixels
    valid_mask = trocar_mask_dilated == 0

    # 3. Remove crop borders
    border_mask = np.zeros((h, w), dtype=bool)

    if h <= 2 * inner_margin_px or w <= 2 * inner_margin_px:
        return None

    border_mask[
        inner_margin_px:h - inner_margin_px,
        inner_margin_px:w - inner_margin_px,
    ] = True

    valid_mask &= border_mask

    # 4. Central elliptical mask
    yy, xx = np.indices((h, w))
    cx = w / 2.0
    cy = h / 2.0

    rx = (w / 2.0) * ellipse_scale
    ry = (h / 2.0) * ellipse_scale

    if rx <= 0 or ry <= 0:
        return None

    ellipse_mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0

    valid_mask &= ellipse_mask

    # 5. Valid XYZ points
    xyz_valid = np.isfinite(xyz_crop).all(axis=2)
    xyz_valid &= np.linalg.norm(xyz_crop, axis=2) > 1e-6

    valid_mask &= xyz_valid

    points = xyz_crop[valid_mask]

    if len(points) == 0:
        return None

    # 6. Depth filtering using the median depth of the candidates
    z_med = np.median(points[:, 2])
    points = points[np.abs(points[:, 2] - z_med) < z_window_mm]

    points = filter_points(points)

    return points


# ============================================================
# Trocar/instrument direction estimation
# ============================================================

def fit_cylinder_axis_fixed_radius(
    points,
    radius_mm=0.7,
    axis_hint=None,
    n_iter=500,
    inlier_dist_mm=0.3,
    min_inliers=10,
    prefilter_axis_factor=5.0,
):
    """
    Fit the visible trocar cylinder with a fixed radius, WITHOUT estimating
    the cylinder direction from the trocar cloud.

    The cylinder direction is forced to be the expected insertion direction
    given by axis_hint, normally:

        axis_hint = c_eye - median(trocar_points)

    Therefore the trocar cloud can only correct the lateral cylinder centre,
    not rotate the axis. This avoids unstable direction estimates when the
    green trocar cloud contains tissue, mask borders or specular outliers.

    Returns:
        p0, axis_dir, inlier_mask, diag
    """
    points = filter_points(points)
    if points is None or len(points) < min_inliers:
        return None, None, None, {
            "cylinder_fit": "not_enough_points",
            "num_points": 0 if points is None else int(len(points)),
        }

    n = len(points)

    # 1. Axis direction: ONLY from the expected insertion direction.
    #    No direction estimation from the trocar cloud is done here.
    if axis_hint is None:
        return None, None, None, {
            "cylinder_fit": "missing_axis_hint",
            "note": "Cylinder fit rejected because no expected insertion axis was provided.",
        }

    axis_dir = np.asarray(axis_hint, dtype=np.float64).reshape(3)
    norm_h = np.linalg.norm(axis_dir)
    if norm_h < 1e-9:
        return None, None, None, {
            "cylinder_fit": "invalid_axis_hint",
            "note": "Cylinder fit rejected because axis_hint has near-zero norm.",
        }
    axis_dir = axis_dir / norm_h

    # 2. Orthonormal basis in the plane perpendicular to the fixed axis.
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(axis_dir, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    e1 = helper - np.dot(helper, axis_dir) * axis_dir
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-9:
        return None, axis_dir, None, {
            "cylinder_fit": "invalid_perpendicular_basis",
        }
    e1 = e1 / e1_norm
    e2 = np.cross(axis_dir, e1)

    # 3. Project points into the fixed-axis coordinate system.
    origin = np.median(points, axis=0)
    pts_c = points - origin
    proj_to_axis = pts_c @ axis_dir
    perp_from_axis = pts_c - np.outer(proj_to_axis, axis_dir)
    dist_from_axis = np.linalg.norm(perp_from_axis, axis=1)
    proj_2d = np.column_stack([pts_c @ e1, pts_c @ e2])

    # 4. Surface-proximity pre-filter.
    #    A valid cylinder-surface point is at distance ≈ radius_mm from the axis.
    #    We keep points in the shell [radius - slack, radius + slack].
    #    slack = prefilter_axis_factor * inlier_dist_mm (default: 5 * 0.3 = 1.5 mm).
    #    This is much tighter than a plain "distance < N*r" tube and removes both
    #    far outliers AND interior points (tissue behind the cylinder surface).
    surface_slack_mm = float(prefilter_axis_factor) * float(inlier_dist_mm)
    prefilter_mask = np.abs(dist_from_axis - radius_mm) < surface_slack_mm
    n_prefiltered = int(prefilter_mask.sum())

    if n_prefiltered >= min_inliers:
        active_idx = np.flatnonzero(prefilter_mask)
    else:
        # Fall back to a loose tube if the surface shell is too empty.
        loose_mask = dist_from_axis < float(prefilter_axis_factor) * float(radius_mm)
        n_loose = int(loose_mask.sum())
        if n_loose >= min_inliers:
            active_idx = np.flatnonzero(loose_mask)
            n_prefiltered = n_loose
        else:
            active_idx = np.arange(n)
            n_prefiltered = n

    proj_2d_ransac = proj_2d[active_idx]
    n_ransac = len(active_idx)

    # 5. Camera-side hint (soft – used as tie-breaker only, NOT as hard filter).
    #    Hard rejection of the near-camera candidate was causing the centre to be
    #    pushed to the wrong side when the visible arc covered more than 90° or
    #    when the cam_perp direction was poorly conditioned.
    #    Instead: when both candidates have the same inlier count we prefer the
    #    one on the FAR side from the camera (dot < 0 with cam_hint_2d).
    use_cam_hint = False
    cam_hint_2d = None
    origin_len = float(np.linalg.norm(origin))
    if origin_len > 1e-6:
        cam_dir_3d = -origin / origin_len
        cam_perp_3d = cam_dir_3d - np.dot(cam_dir_3d, axis_dir) * axis_dir
        cam_perp_len = float(np.linalg.norm(cam_perp_3d))
        if cam_perp_len > 1e-4:
            cam_perp_3d = cam_perp_3d / cam_perp_len
            cam_hint_2d = np.array([
                float(cam_perp_3d @ e1),
                float(cam_perp_3d @ e2),
            ], dtype=np.float64)
            use_cam_hint = True

    # 6. RANSAC in 2-D, using only pre-filtered points for sampling and scoring.
    rng = np.random.default_rng(42)
    best_center_2d = None
    best_inlier_mask_active = None
    best_score = -1
    best_median_residual = None

    if n_ransac < 2:
        return None, axis_dir, None, {
            "cylinder_fit": "not_enough_prefiltered_points",
            "num_points": int(n),
            "num_prefiltered_points": int(n_prefiltered),
            "surface_slack_mm": float(surface_slack_mm),
            "cam_hint_active": bool(use_cam_hint),
        }

    for _ in range(n_iter):
        i, j = rng.choice(n_ransac, 2, replace=False)
        q1, q2 = proj_2d_ransac[i], proj_2d_ransac[j]

        seg = q2 - q1
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6 or seg_len > 2.0 * radius_mm:
            continue

        mid = 0.5 * (q1 + q2)
        h_sq = radius_mm ** 2 - (0.5 * seg_len) ** 2
        if h_sq < 0.0:
            continue

        h = np.sqrt(h_sq)
        perp = np.array([-seg[1], seg[0]], dtype=np.float64) / seg_len

        for sign in (1.0, -1.0):
            c = mid + sign * h * perp

            residuals_active = np.abs(
                np.linalg.norm(proj_2d_ransac - c[None, :], axis=1) - radius_mm
            )
            mask_active = residuals_active < inlier_dist_mm
            score = int(mask_active.sum())

            if score <= 0:
                continue

            median_res = float(np.median(residuals_active[mask_active]))

            # Cam-hint soft tie-breaker: slightly prefer the far-side candidate.
            # A small bonus avoids the spurious centre winning a tie without
            # hard-rejecting it when the arc geometry is ambiguous.
            if use_cam_hint:
                cam_dot = float(np.dot(c, cam_hint_2d))
                # far-side (dot < 0) gets +0.5 bonus; near-side gets -0.5 bonus
                score_eff = score - 0.5 * float(np.sign(cam_dot))
            else:
                score_eff = float(score)

            if (
                score_eff > (best_score if best_median_residual is None else best_score)
                or (abs(score_eff - best_score) < 0.6 and (best_median_residual is None or median_res < best_median_residual))
            ):
                best_score = score_eff
                best_center_2d = c.copy()
                best_inlier_mask_active = mask_active.copy()
                best_median_residual = median_res

    if best_center_2d is None or best_score < min_inliers:
        return None, axis_dir, None, {
            "cylinder_fit": "ransac_no_consensus",
            "best_inlier_count": int(best_score) if best_score >= 0 else 0,
            "num_points": int(n),
            "num_prefiltered_points": int(n_prefiltered),
            "surface_slack_mm": float(surface_slack_mm),
            "cam_hint_active": bool(use_cam_hint),
            "axis_source": "axis_hint_only_no_cloud_direction_fit",
        }

    # Convert active inlier mask back to full point-cloud mask.
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_inlier_mask[active_idx[best_inlier_mask_active]] = True

    inlier_pts = points[best_inlier_mask]

    # 7. Axis reference and visible midpoint.
    #    The axis direction remains fixed. No axis refinement is performed.
    axis_ref_3d = origin + best_center_2d[0] * e1 + best_center_2d[1] * e2
    inlier_heights = (inlier_pts - axis_ref_3d) @ axis_dir
    h_min = float(inlier_heights.min())
    h_max = float(inlier_heights.max())
    p0 = axis_ref_3d + 0.5 * (h_min + h_max) * axis_dir

    # 8. Diagnostics.
    pts_from_ref = points - axis_ref_3d
    along_ax = pts_from_ref @ axis_dir
    perp_vecs = pts_from_ref - np.outer(along_ax, axis_dir)
    dist_surf = np.abs(np.linalg.norm(perp_vecs, axis=1) - radius_mm)
    centroid_raw = np.median(points, axis=0)

    p_start = axis_ref_3d + h_min * axis_dir
    p_end = axis_ref_3d + h_max * axis_dir

    diag = {
        "cylinder_fit": "success",
        "axis_source": "axis_hint_only_no_cloud_direction_fit",
        "axis_refinement": "disabled",
        "num_points": int(n),
        "num_prefiltered_points": int(n_prefiltered),
        "surface_slack_mm": float(surface_slack_mm),
        "num_cylinder_inliers": int(best_score),
        "cylinder_visible_height_mm": float(h_max - h_min),
        "cylinder_radius_fixed_mm": float(radius_mm),
        "median_surface_residual_mm": float(np.median(dist_surf[best_inlier_mask])),
        "p0_offset_from_centroid_mm": float(np.linalg.norm(p0 - centroid_raw)),
        "cam_hint_active": bool(use_cam_hint),
        "cylinder_axis_dir": axis_dir.tolist(),
        "cylinder_axis_point_mm": axis_ref_3d.tolist(),
        "cylinder_midpoint_mm": p0.tolist(),
        "cylinder_p_start_mm": p_start.tolist(),
        "cylinder_p_end_mm": p_end.tolist(),
    }

    return p0, axis_dir, best_inlier_mask, diag



def keep_largest_component(mask_u8):
    """Keep only the largest connected component of a binary mask."""
    if mask_u8 is None:
        return None

    mask = (mask_u8 > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    # Ignore background label 0.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    return (labels == best_label).astype(np.uint8)


def clean_trocar_mask_for_center(trocar_mask, open_px=0, close_px=3):
    """
    Clean the trocar mask before computing its 2D center.

    We keep the largest component and optionally close small holes. This is used
    only to estimate the projected lateral center of the trocar, not to extract
    the eye points.
    """
    if trocar_mask is None:
        return None

    mask = (trocar_mask > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return None

    mask = keep_largest_component(mask)
    if mask is None or int(mask.sum()) == 0:
        return None

    if close_px and close_px > 1:
        k = np.ones((int(close_px), int(close_px)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        mask = keep_largest_component(mask)

    if open_px and open_px > 1:
        k = np.ones((int(open_px), int(open_px)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = keep_largest_component(mask)

    return mask


def mask_centroid_2d(mask_u8):
    """Return subpixel centroid (u, v) of a binary mask in image coordinates."""
    if mask_u8 is None:
        return None

    mask = (mask_u8 > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return None

    M = cv2.moments(mask, binaryImage=True)
    if abs(M["m00"]) < 1e-9:
        return None

    u = float(M["m10"] / M["m00"])
    v = float(M["m01"] / M["m00"])
    return np.array([u, v], dtype=np.float64)


def backproject_mask_center_with_depth(
    xyz_crop,
    center_uv,
    depth_z_mm,
    search_radii_px=(2, 4, 6, 10, 15, 25),
):
    """
    Reconstruct a 3D point from a 2D mask center and a robust depth value.

    We do not need the camera intrinsics explicitly because xyz_crop is an
    organized point cloud. Around the requested pixel we estimate the local ray
    direction as (X/Z, Y/Z, 1), then place the point at depth_z_mm.
    """
    if xyz_crop is None or center_uv is None or depth_z_mm is None:
        return None, None

    if not np.isfinite(depth_z_mm) or abs(float(depth_z_mm)) < 1e-9:
        return None, None

    h, w = xyz_crop.shape[:2]
    u, v = float(center_uv[0]), float(center_uv[1])

    if not (0 <= u < w and 0 <= v < h):
        return None, None

    uu = int(round(u))
    vv = int(round(v))

    for radius in search_radii_px:
        r = int(radius)
        x1 = max(0, uu - r)
        x2 = min(w, uu + r + 1)
        y1 = max(0, vv - r)
        y2 = min(h, vv + r + 1)

        patch = xyz_crop[y1:y2, x1:x2].reshape(-1, 3).astype(np.float64)
        valid = np.isfinite(patch).all(axis=1)
        valid &= np.linalg.norm(patch, axis=1) > 1e-6
        valid &= np.abs(patch[:, 2]) > 1e-9
        patch = patch[valid]

        if len(patch) == 0:
            continue

        # Estimate local ray direction robustly from organized XYZ.
        rays_xy = patch[:, :2] / patch[:, 2:3]
        ray_x = float(np.median(rays_xy[:, 0]))
        ray_y = float(np.median(rays_xy[:, 1]))

        p = np.array([
            ray_x * float(depth_z_mm),
            ray_y * float(depth_z_mm),
            float(depth_z_mm),
        ], dtype=np.float64)

        diag = {
            "ray_search_radius_px": int(r),
            "num_local_ray_points": int(len(patch)),
            "local_ray_xy": [ray_x, ray_y],
            "center_uv_rounded_px": [int(uu), int(vv)],
            "local_ray_patch_xyxy": [int(x1), int(y1), int(x2 - 1), int(y2 - 1)],
        }
        return p, diag

    return None, None


def estimate_p0_from_mask2d_depth(
    xyz_crop,
    trocar_mask,
    trocar_points,
):
    """
    Estimate p0_trocar using 2D lateral position + 3D median depth.

    Motivation:
      - The 3D point cloud of a cylindrical trocar only contains the visible
        surface and is therefore laterally biased.
      - The 2D mask gives a better estimate of the projected lateral center of
        the trocar silhouette.
      - The organized XYZ cloud still provides the depth scale.

    Returns:
        p0, diag
    """
    points = filter_points(trocar_points)
    if points is None or len(points) < 5:
        return None, {
            "p0_method": "mask2d_depth_failed_not_enough_trocar_points",
            "num_filtered_trocar_points": 0 if points is None else int(len(points)),
        }

    raw_mask = (trocar_mask > 0).astype(np.uint8) if trocar_mask is not None else None
    num_mask_pixels_raw = int(raw_mask.sum()) if raw_mask is not None else 0

    clean_mask = clean_trocar_mask_for_center(trocar_mask)
    center_uv = mask_centroid_2d(clean_mask)
    if center_uv is None:
        return None, {
            "p0_method": "mask2d_depth_failed_empty_mask",
            "num_mask_pixels_raw": num_mask_pixels_raw,
            "num_filtered_trocar_points": int(len(points)),
        }

    ys_clean, xs_clean = np.where(clean_mask > 0)
    clean_bbox_xyxy = [
        int(xs_clean.min()),
        int(ys_clean.min()),
        int(xs_clean.max()),
        int(ys_clean.max()),
    ] if len(xs_clean) > 0 else None

    z_med = float(np.median(points[:, 2]))
    p0, ray_diag = backproject_mask_center_with_depth(
        xyz_crop=xyz_crop,
        center_uv=center_uv,
        depth_z_mm=z_med,
    )

    if p0 is None:
        return None, {
            "p0_method": "mask2d_depth_failed_no_local_ray",
            "mask_center_uv": center_uv.tolist(),
            "depth_z_median_mm": z_med,
            "num_filtered_trocar_points": int(len(points)),
        }

    centroid_3d = np.median(points, axis=0)
    num_mask_pixels_clean = int(clean_mask.sum()) if clean_mask is not None else 0

    diag = {
        "p0_method": "mask2d_centroid_depth_median",
        "mask_center_uv": center_uv.tolist(),
        "mask_center_uv_rounded_px": [int(round(center_uv[0])), int(round(center_uv[1]))],
        "depth_z_median_mm": z_med,
        "num_mask_pixels_raw": num_mask_pixels_raw,
        "num_mask_pixels_clean": num_mask_pixels_clean,
        "clean_mask_area_ratio": float(num_mask_pixels_clean / max(num_mask_pixels_raw, 1)),
        "clean_mask_bbox_xyxy": clean_bbox_xyxy,
        "num_filtered_trocar_points": int(len(points)),
        "p0_trocar_mask2d_depth_mm": p0.tolist(),
        "p0_trocar_median3d_mm": centroid_3d.tolist(),
        "p0_shift_from_median3d_mm": float(np.linalg.norm(p0 - centroid_3d)),
        "note": (
            "p0 is reconstructed from the 2D centroid of the cleaned trocar mask "
            "and the median Z depth of the filtered trocar point cloud."
        ),
    }
    if ray_diag is not None:
        diag.update(ray_diag)

    return p0, diag


def build_visual_trocar_cylinder_outside_eye(
    p_entry_eye,
    v_out,
    length_mm,
    radius_mm=1.0,
):
    """
    Build a physical/visual cylinder only outside the eye sphere.

    The cylinder starts at the line/sphere intersection p_entry_eye and extends
    along v_out, i.e. away from the fitted eye center. This avoids drawing the
    cylinder inside the eye sphere in debug PLY files.

    This is not a free 3D cylinder fit. It is only a known-radius visualization
    of the planned trocar/insertion axis.
    """
    if p_entry_eye is None or v_out is None:
        return None

    p_start = np.asarray(p_entry_eye, dtype=np.float64).reshape(3)
    axis_dir = np.asarray(v_out, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis_dir))
    if n < 1e-9:
        return None
    axis_dir = axis_dir / n

    length_mm = float(length_mm)
    if not np.isfinite(length_mm) or length_mm <= 0.0:
        return None

    p_end = p_start + length_mm * axis_dir
    midpoint = 0.5 * (p_start + p_end)

    return {
        "midpoint_mm": midpoint,
        "axis_dir": axis_dir,
        "radius_mm": float(radius_mm),
        "visible_height_mm": float(length_mm),
        "p_start_mm": p_start,
        "p_end_mm": p_end,
        "source": "visual_cylinder_from_entry_eye_outside",
        "note": "Physical-radius visualization only; cylinder starts at p_entry_eye_mm and extends only to p_entry_trocar_mm, outside the fitted eye sphere.",
    }


def fit_trocar_axis_to_eye_center(
    trocar_points,
    c_eye,
    xyz_crop=None,
    trocar_mask=None,
    cylinder_radius_mm=0.5,
    cylinder_n_iter=500,
    cylinder_inlier_dist_mm=0.3,
    cylinder_min_inliers=10,
    cylinder_prefilter_axis_factor=3.0,
    max_axis_angle_deg=25.0,
):
    """
    Estimate p0 and v using mask-2D lateral center + median 3D depth.

    This deliberately avoids fitting a full 3D cylinder to the trocar cloud,
    because the organized 3D cloud contains only a partial visible surface of
    the cylinder and can be laterally biased.

    Main method:
        p0 = backproject(centroid_2D(mask), median_Z(trocar_points))
        v  = normalize(c_eye - p0)

    Fallback:
        If the mask/depth reconstruction fails, use the robust 3D median of the
        filtered trocar points.
    """
    points = filter_points(trocar_points)
    if points is None or len(points) < 5:
        return None, None, None, None

    centroid_3d = np.median(points, axis=0)

    p0_mask, mask_diag = estimate_p0_from_mask2d_depth(
        xyz_crop=xyz_crop,
        trocar_mask=trocar_mask,
        trocar_points=points,
    )

    if p0_mask is not None:
        p0 = p0_mask
        method = "mask2d_centroid_depth_median"
    else:
        p0 = centroid_3d
        method = "median3d_fallback_after_mask2d_depth_fail"

    # Final insertion direction is defined from the estimated trocar center
    # to the fitted eye sphere center.
    v = c_eye - p0
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        return None, None, None, None
    v = orient_axis_towards_target(p0, v / norm_v, c_eye)

    # Optional sanity direction from raw 3D median to eye center.
    axis_to_eye_from_median = c_eye - centroid_3d
    n_med = float(np.linalg.norm(axis_to_eye_from_median))
    angle_mask_vs_median_deg = None
    if n_med > 1e-9:
        axis_to_eye_from_median = axis_to_eye_from_median / n_med
        cos_a = float(np.clip(np.dot(v, axis_to_eye_from_median), -1.0, 1.0))
        angle_mask_vs_median_deg = float(np.degrees(np.arccos(cos_a)))

    extent_x = float(np.ptp(points[:, 0]))
    extent_y = float(np.ptp(points[:, 1]))
    extent_z = float(np.ptp(points[:, 2]))

    diag = {
        "axis_method": method,
        "p0_estimation": mask_diag,
        "extent_x_mm": extent_x,
        "extent_y_mm": extent_y,
        "extent_z_mm": extent_z,
        "max_extent_mm": float(max(extent_x, extent_y, extent_z)),
        "num_filtered_trocar_points": int(len(points)),
        "p0_trocar_median3d_mm": centroid_3d.tolist(),
        "p0_trocar_used_mm": p0.tolist(),
        "p0_shift_from_median3d_mm": float(np.linalg.norm(p0 - centroid_3d)),
        "angle_mask_axis_vs_median3d_axis_deg": angle_mask_vs_median_deg,
        "warning_mask_axis_differs_from_median3d_axis": bool(
            angle_mask_vs_median_deg is not None and angle_mask_vs_median_deg > 15.0
        ),
        "cylinder": {
            "cylinder_fit": "not_run_visual_model_only",
            "note": "The trocar is modeled as a physical cylinder for visualization, but p0 is estimated from mask 2D center and median depth.",
        },
    }

    return p0, v, method, diag

def fit_trocar_axis_to_target(trocar_points, p_target):
    """
    Backward-compatible generic helper.

    Prefer fit_trocar_axis_to_eye_center(...) in this script. This function is
    kept only in case another script imports it directly.
    """
    points = filter_points(trocar_points)

    if points is None or len(points) < 5:
        return None, None, None, None

    p0 = np.median(points, axis=0)

    v = p_target - p0
    norm_v = np.linalg.norm(v)

    if norm_v < 1e-9:
        return None, None, None, None

    v = v / norm_v
    v = orient_axis_towards_target(p0, v, p_target)

    extent_x = np.ptp(points[:, 0])
    extent_y = np.ptp(points[:, 1])
    extent_z = np.ptp(points[:, 2])
    max_extent = max(extent_x, extent_y, extent_z)

    method = "centroid_to_target"

    diag = {
        "axis_method": method,
        "extent_x_mm": float(extent_x),
        "extent_y_mm": float(extent_y),
        "extent_z_mm": float(extent_z),
        "max_extent_mm": float(max_extent),
        "num_filtered_trocar_points": int(len(points)),
        "note": (
            "Generic target helper. In estimate_trocar_pose this is not used; "
            "the axis is computed towards the fitted eye sphere center."
        ),
    }

    return p0, v, method, diag


# ============================================================
# Main pose estimation function
# ============================================================

def estimate_trocar_pose(
    xyz_crop,
    trocar_mask,
    trocar_points,
    trocar_entry_offset_mm=DEFAULT_TROCAR_ENTRY_OFFSET_MM,
    target_clearance_mm=DEFAULT_TARGET_CLEARANCE_MM,
    trocar_cylinder_radius_mm=1.0,
):
    """
    Estimate:
        - fitted eye sphere
        - trocar cylinder-center / centroid fallback
        - insertion vector from p0_trocar to eye sphere center
        - eye entry point on the fitted sphere
        - trocar-side point outside the eye surface
        - pre-entry / approach point in camera frame
        - insertion z-axis in camera frame

    This function deliberately does NOT build x/y axes and does NOT build a
    6DoF pose matrix. The roll around the trocar axis is not observable from
    the trocar geometry, so it is computed later in robot frame.

    Important:
        The direction IS now defined using the fitted eye sphere center:

            v = c_eye - p0_trocar

        where p0_trocar is the fitted cylinder-axis midpoint when possible,
        with robust visible-trocar centroid as fallback.
    """
    if xyz_crop is None or trocar_mask is None:
        return None

    if xyz_crop.shape[:2] != trocar_mask.shape[:2]:
        raise ValueError(
            f"xyz_crop and trocar_mask are not aligned: "
            f"xyz_crop={xyz_crop.shape[:2]}, trocar_mask={trocar_mask.shape[:2]}"
        )

    if trocar_points is None or len(trocar_points) < 5:
        return None

    # 1. Extract a clean eye point cloud
    eye_points = extract_eye_points_for_sphere(
        xyz_crop=xyz_crop,
        trocar_mask=trocar_mask,
        trocar_points=trocar_points,
    )

    if eye_points is None or len(eye_points) < 200:
        return None

    # 2. Fit the eye sphere
    c_eye, r_eye, sphere_inliers = fit_sphere_ransac(
        eye_points,
        n_iter=1500,
        distance_threshold=0.8,
        radius_range=(13.5, 15.5),
        min_inliers=200,
    )

    if c_eye is None or r_eye is None or sphere_inliers is None:
        return None

    # 3. Estimate insertion direction from trocar cylinder midpoint to eye sphere center.
    p0, v, axis_method, axis_diag = fit_trocar_axis_to_eye_center(
        trocar_points=trocar_points,
        c_eye=c_eye,
        xyz_crop=xyz_crop,
        trocar_mask=trocar_mask,
        cylinder_radius_mm=trocar_cylinder_radius_mm,
    )

    if p0 is None or v is None:
        return None

    # 4. Line-sphere intersection.
    #    This is the estimated anatomical entry point on the eye surface for
    #    the line joining p0_trocar and the eye sphere center.
    p_entry_eye = intersect_line_sphere(
        p0=p0,
        v=v,
        center=c_eye,
        radius=r_eye,
    )

    if p_entry_eye is None:
        return None

    # 5. Define only the geometric line needed for the robot step.
    #    NO x/y axes and NO rotation matrix are created here.
    #
    #    v_in points towards the eye sphere center / inside the eye.
    #    v_out points away from the eye along the same line.
    v_in = v
    v_out = -v_in

    # Point just outside the eye surface, along the trocar/instrument axis.
    p_entry_trocar = p_entry_eye + trocar_entry_offset_mm * v_out

    # Pre-entry / approach point outside the eye.
    # This is only a 3D point in camera frame.
    p_pre = p_entry_trocar + target_clearance_mm * v_out

    # Diagnostic: distance between eye center and planned axis.
    # This should be close to zero because the axis is constructed towards
    # the fitted sphere center.
    distance_center_to_axis = np.linalg.norm(np.cross(c_eye - p0, v))

    # Build only the short physical/visual trocar cylinder outside the fitted
    # eye sphere. It starts exactly at p_entry_eye and ends at p_entry_trocar.
    # Therefore its height is trocar_entry_offset_mm (default: 2 mm), NOT the
    # full 50 mm approach clearance.
    visual_cylinder_length_mm = float(trocar_entry_offset_mm)
    trocar_cylinder = build_visual_trocar_cylinder_outside_eye(
        p_entry_eye=p_entry_eye,
        v_out=v_out,
        length_mm=visual_cylinder_length_mm,
        radius_mm=trocar_cylinder_radius_mm,
    )

    return {
        "p0_trocar_mm": p0,
        "v_trocar": v,        # kept as alias
        "v_in_cam": v_in,
        "z_axis_cam": v_in,
        "axis_method": axis_method,
        "axis_diag": axis_diag,
        "p0_estimation_debug": (
            axis_diag.get("p0_estimation")
            if isinstance(axis_diag, dict) else None
        ),
        "trocar_cylinder": trocar_cylinder,

        "eye_center_mm": c_eye,
        "eye_radius_mm": float(r_eye),


        "p_entry_eye_mm": p_entry_eye,
        "p_entry_trocar_mm": p_entry_trocar,
        "p_pre_mm": p_pre,
        "p_approach_cam_mm": p_pre,
        "p_target_mm": p_pre,  # kept as alias for backward compatibility

        "trocar_entry_offset_mm": float(trocar_entry_offset_mm),
        "target_clearance_mm": float(target_clearance_mm),

        "distance_eye_center_to_axis_mm": float(distance_center_to_axis),

        "num_trocar_points": int(len(trocar_points)),
        "num_eye_points": int(len(eye_points)),
        "num_sphere_inliers": int(len(sphere_inliers)) if sphere_inliers is not None else 0,

        "eye_points_used": eye_points,
        "sphere_inliers": sphere_inliers,
    }
