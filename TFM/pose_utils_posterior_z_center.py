import numpy as np
import cv2


DEFAULT_TROCAR_ENTRY_OFFSET_MM = 1.0
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

    Returns the first intersection found when moving from p0 along v.
    In this script, v should point from the visible trocar centroid towards
    the fitted eye sphere center.
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

    candidates = [(t1, p1), (t2, p2)]

    # Prefer the first positive intersection along the insertion direction.
    positive = [(t, p) for t, p in candidates if t >= 0]

    if positive:
        _, p_entry = min(positive, key=lambda x: x[0])
    else:
        _, p_entry = min(candidates, key=lambda x: abs(x[0]))

    return p_entry


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
# Posterior pole estimation
# ============================================================

def estimate_posterior_pole_from_visible_eye(eye_points, c_eye, r_eye):
    """
    Estimate the posterior pole from the visible eye surface.

    Idea:
        1. Use the robust centroid of the visible eye points.
        2. Define the visible/anterior direction from the sphere center towards
           that visible centroid.
        3. The posterior pole is approximated as the opposite point on the
           fitted sphere.

    This is a geometric approximation. If the phantom has anatomical markers
    or if the true eye pose is available, use that information instead.

    Returns:
        p_posterior, p_visible, n_anterior
    """
    points = filter_points(eye_points)

    if points is None or len(points) < 20:
        return None, None, None

    p_visible = np.median(points, axis=0)

    n_anterior = p_visible - c_eye
    norm_n = np.linalg.norm(n_anterior)

    if norm_n < 1e-9:
        return None, None, None

    n_anterior = n_anterior / norm_n

    # Opposite point on the sphere.
    p_posterior = c_eye - r_eye * n_anterior

    return p_posterior, p_visible, n_anterior


# ============================================================
# Trocar/instrument direction estimation
# ============================================================

def fit_trocar_axis_to_eye_center(trocar_points, c_eye):
    """
    Estimate the insertion direction using the visible trocar centroid as origin
    and the fitted eye sphere center as destination.

    That is:
        p0 = robust centroid of the visible trocar
        v  = direction from p0 towards c_eye

    This is the radial/geometric version:
        visible trocar centroid -> eye sphere center
    """
    points = filter_points(trocar_points)

    if points is None or len(points) < 5:
        return None, None, None, None

    p0 = np.median(points, axis=0)

    v = c_eye - p0
    norm_v = np.linalg.norm(v)

    if norm_v < 1e-9:
        return None, None, None, None

    v = v / norm_v
    v = orient_axis_towards_target(p0, v, c_eye)

    extent_x = np.ptp(points[:, 0])
    extent_y = np.ptp(points[:, 1])
    extent_z = np.ptp(points[:, 2])
    max_extent = max(extent_x, extent_y, extent_z)

    method = "centroid_to_eye_center"

    diag = {
        "axis_method": method,
        "extent_x_mm": float(extent_x),
        "extent_y_mm": float(extent_y),
        "extent_z_mm": float(extent_z),
        "max_extent_mm": float(max_extent),
        "num_filtered_trocar_points": int(len(points)),
        "note": (
            "PCA disabled. Axis estimated as the direction from "
            "the visible trocar centroid to the fitted eye sphere center."
        ),
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
):
    """
    Estimate:
        - visible trocar centroid
        - eye sphere
        - optional estimated posterior pole for diagnostics
        - insertion vector from visible trocar centroid to eye sphere center
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

        where p0_trocar is the robust centroid of the visible trocar points.
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

    # 3. Estimate posterior pole only as optional diagnostic information.
    #    It is NOT used to compute the insertion vector in this version.
    p_posterior, p_visible_eye, n_anterior = estimate_posterior_pole_from_visible_eye(
        eye_points=sphere_inliers,
        c_eye=c_eye,
        r_eye=r_eye,
    )

    # 4. Estimate insertion direction from trocar centroid to eye sphere center.
    p0, v, axis_method, axis_diag = fit_trocar_axis_to_eye_center(
        trocar_points=trocar_points,
        c_eye=c_eye,
    )

    if p0 is None or v is None:
        return None

    # 5. Line-sphere intersection.
    #    This is the estimated anatomical entry point on the eye surface for
    #    the line joining the visible trocar centroid and the eye sphere center.
    p_entry_eye = intersect_line_sphere(
        p0=p0,
        v=v,
        center=c_eye,
        radius=r_eye,
    )

    if p_entry_eye is None:
        return None

    # 6. Define only the geometric line needed for the robot step.
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

    return {
        "p0_trocar_mm": p0,
        "v_trocar": v,        # kept as alias
        "v_in_cam": v_in,
        "z_axis_cam": v_in,
        "axis_method": axis_method,
        "axis_diag": axis_diag,

        "eye_center_mm": c_eye,
        "eye_radius_mm": float(r_eye),

        "p_visible_eye_mm": p_visible_eye,
        "n_anterior": n_anterior,
        "p_posterior_pole_mm": p_posterior,

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
