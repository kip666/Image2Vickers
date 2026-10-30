"""
Vickers Microhardness Indentation Analysis Module.
Detects indentation edges under any rotation angle (square, diamond, arbitrary),
robust against defocus, scratches, polishing artifacts, and surface dirt.
Provides interactive mouse corner dragging with real-time recalculation
and user-defined microhardness reference calibration.
"""

import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

DEFAULT_CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"


def calculate_vickers_hardness(d_avg_um, load_gf=300.0):
    """
    Calculates Vickers microhardness (HV) according to ISO 6507 / ASTM E384 standards.
    Formula:
        HV = 2 * F * sin(136° / 2) / d^2 ≈ 1.85437 * F_kgf / (d_mm^2)
    """
    if d_avg_um <= 0:
        return 0.0
    load_kgf = float(load_gf) / 1000.0
    d_mm = float(d_avg_um) / 1000.0
    hv = 1.85437 * load_kgf / (d_mm ** 2)
    return float(hv)


def calculate_diagonal_from_hardness(hv, load_gf=300.0):
    """
    Calculates expected theoretical indentation diagonal length in micrometers (um)
    from target Vickers hardness (HV) and test load (gf).
    Formula:
        HV = 1.85437 * F_kgf / (d_mm^2)
        => d_mm = sqrt(1.85437 * F_kgf / HV)
        => d_um = d_mm * 1000.0
    """
    if hv <= 0 or load_gf <= 0:
        return 0.0
    load_kgf = float(load_gf) / 1000.0
    d_mm = np.sqrt(1.85437 * load_kgf / float(hv))
    d_um = float(d_mm * 1000.0)
    return d_um


def prompt_target_hardness(initial_val=500.0, load_gf=300.0):
    """
    Prompts the user to enter certified reference hardness (HV).
    Uses Tkinter modal dialog with fallback to console input.
    """
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        val = simpledialog.askfloat(
            "Vickers Calibration",
            f"Enter certified reference hardness (HV) for test block:\n(Test load: {int(load_gf)} gf / {load_gf/1000.0:.3f} kgf)",
            initialvalue=float(initial_val),
            minvalue=1.0,
            maxvalue=5000.0,
            parent=root
        )
        root.destroy()
        return val
    except Exception:
        try:
            raw = input(f"Enter reference hardness in HV [{initial_val}]: ").strip()
            if not raw:
                return float(initial_val)
            return float(raw)
        except Exception:
            return float(initial_val)


def _estimate_background(gray):
    """
    Estimates smooth background illumination on a downsampled image.
    A large effective kernel (~360 px) preserves local contrast
    even for large and heavily defocused indentations.
    """
    h, w = gray.shape
    scale = 8
    small = cv2.resize(gray, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    bg_small = cv2.GaussianBlur(small, (45, 45), 0)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return bg


def _find_indent_core(darkness):
    """
    Locates the center and reference inscribed radius of the pyramid indentation,
    filtering out adjacent dirt specks, scratches, and noise.
    Supports both sharp and defocused frames.
    """
    dark_smooth = cv2.GaussianBlur(darkness, (9, 9), 2.0)
    max_val = float(np.max(dark_smooth))
    if max_val < 5.0:
        return None

    h_dark, w_dark = darkness.shape
    border_margin = 15

    best_candidate = None
    best_quality = -1.0

    for ratio in (0.48, 0.38, 0.28, 0.20):
        thr = max(4.0, max_val * ratio)
        _, mask = cv2.threshold(dark_smooth, thr, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        # Close the central specular reflection
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        # Remove thin scratch tails and small dust
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 2200 or area > 150000:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx <= border_margin or by <= border_margin or
                    bx + bw >= w_dark - border_margin or by + bh >= h_dark - border_margin):
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area

            c_mask = np.zeros_like(opened)
            cv2.drawContours(c_mask, [hull], -1, 255, -1)
            dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
            _, r_inscribed, _, max_loc = cv2.minMaxLoc(dist)

            if r_inscribed < 22.0 or r_inscribed > 220.0:
                continue

            expected_sq_area = 4.0 * (r_inscribed ** 2)
            compactness = min(hull_area, expected_sq_area) / max(hull_area, expected_sq_area)

            peak = (dist >= r_inscribed * 0.75).astype(np.uint8)
            M = cv2.moments(peak)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = float(max_loc[0]), float(max_loc[1])

            core_darkness = float(cv2.mean(dark_smooth, mask=c_mask)[0])
            quality = (core_darkness / max_val) * (solidity ** 1.0) * (compactness ** 1.2) * min(1.0, r_inscribed / 60.0)

            if quality > best_quality:
                best_quality = quality
                best_candidate = (cx, cy, float(r_inscribed))

        if best_candidate is not None and best_quality > 0.35:
            break

    return best_candidate


def _fit_line_ransac_2d(pts, weights, expected_dir, max_angle_deg=18.0):
    """
    Robust RANSAC line fitting (vx, vy, x0, y0) constrained by expected direction.
    """
    if len(pts) < 8:
        return None

    pts_arr = np.asarray(pts, dtype=np.float32)
    w_arr = np.asarray(weights, dtype=np.float32)
    w_sum = float(np.sum(w_arr))
    if w_sum <= 1e-8:
        return None
    w_prob = w_arr / w_sum

    ex, ey = float(expected_dir[0]), float(expected_dir[1])
    cos_min = np.cos(np.deg2rad(max_angle_deg))

    n_pts = len(pts_arr)
    rng = np.random.default_rng(12345)
    best_inliers = None
    best_score = -1.0

    for _ in range(250):
        idx = rng.choice(n_pts, size=2, replace=False, p=w_prob)
        p1, p2 = pts_arr[idx[0]], pts_arr[idx[1]]
        d = p2 - p1
        norm = float(np.hypot(d[0], d[1]))
        if norm < 10.0:
            continue
        vx, vy = d[0] / norm, d[1] / norm

        if abs(vx * ex + vy * ey) < cos_min:
            continue

        dist = np.abs((pts_arr[:, 0] - p1[0]) * vy - (pts_arr[:, 1] - p1[1]) * vx)
        inliers = dist < 2.5
        score = float(np.sum(w_arr[inliers]))

        if score > best_score and np.sum(inliers) >= 8:
            best_score = score
            best_inliers = inliers

    inlier_pts = pts_arr[best_inliers] if best_inliers is not None else pts_arr
    line = cv2.fitLine(inlier_pts, cv2.DIST_L12, 0, 0.01, 0.01)
    return float(line[0][0]), float(line[1][0]), float(line[2][0]), float(line[3][0])


def detect_indentation(image_bgr):
    """
    Universal Vickers indentation detection:
    - Supports arbitrary orientation (0° square, 45° diamond, tilted)
    - Robust against defocus via adaptive gradient scanning
    - Suppresses surface scratches and dirt via core localization and normal gradient filtering
    """
    if image_bgr is None:
        return {'success': False, 'error': 'Empty image'}

    if len(image_bgr.shape) == 3:
        gray = image_bgr[:, :, 1].copy()
    else:
        gray = image_bgr.copy()

    h_img, w_img = gray.shape

    # 1. Background subtraction
    bg = _estimate_background(gray)
    darkness = cv2.subtract(bg, gray)

    # 2. Localize core
    core = _find_indent_core(darkness)
    if core is None:
        return {'success': False, 'error': 'Indent not found in frame'}

    cx_g, cy_g, r_in = core
    half_side = float(np.clip(r_in * 1.04, 30.0, 240.0))

    roi_rad = int(half_side * 1.95)
    rx1 = max(0, int(cx_g - roi_rad))
    ry1 = max(0, int(cy_g - roi_rad))
    rx2 = min(w_img, int(cx_g + roi_rad))
    ry2 = min(h_img, int(cy_g + roi_rad))

    roi_gray = gray[ry1:ry2, rx1:rx2]
    rh, rw = roi_gray.shape
    cx = cx_g - rx1
    cy = cy_g - ry1

    # 3. Brightness gradients
    roi_blur = cv2.GaussianBlur(roi_gray, (7, 7), 1.6)
    gx = cv2.Sobel(roi_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_blur, cv2.CV_32F, 0, 1, ksize=3)
    g_mag = np.hypot(gx, gy)

    # 4. Rotation angle detection
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    dx_c = xx - cx
    dy_c = yy - cy
    dist_c = np.hypot(dx_c, dy_c) + 1e-6

    radial_grad = (gx * dx_c + gy * dy_c) / dist_c
    ring_mask = (dist_c >= half_side * 0.60) & (dist_c <= half_side * 1.40) & (radial_grad > 2.0)

    if np.count_nonzero(ring_mask) < 50:
        return {'success': False, 'error': 'Insufficient edge contrast (heavy defocus)'}

    grad_angles = np.arctan2(gy[ring_mask], gx[ring_mask])
    weights_ring = radial_grad[ring_mask]

    mod_angles = np.mod(grad_angles, np.pi / 2.0)
    n_bins = 180
    hist, bin_edges = np.histogram(mod_angles, bins=n_bins, range=(0.0, np.pi / 2.0), weights=weights_ring)
    hist_pad = np.concatenate([hist[-7:], hist, hist[:7]])
    kernel = np.hanning(15)
    kernel /= kernel.sum()
    hist_smooth = np.convolve(hist_pad, kernel, mode='valid')
    best_bin = int(np.argmax(hist_smooth))
    alpha0 = float((bin_edges[best_bin] + bin_edges[best_bin + 1]) * 0.5)

    normals = []
    for k in range(4):
        ang = alpha0 + k * (np.pi / 2.0)
        normals.append((float(np.cos(ang)), float(np.sin(ang)), ang))

    # 5. Radial scanning along rotated coordinates
    tangential_span = int(half_side * 0.68)
    r_min = int(half_side * 0.58)
    r_max = int(half_side * 1.45)
    sigma_r = half_side * 0.25

    edge_lines = []
    edge_weights_sum = []
    edge_pts_list = []

    for nx, ny, ang in normals:
        tx, ty = -ny, nx

        pts_side = []
        w_side = []

        for s in range(-tangential_span, tangential_span + 1):
            r_vals = np.arange(r_min, r_max + 1, dtype=np.float32)
            px = cx + s * tx + r_vals * nx
            py = cy + s * ty + r_vals * ny

            ix = np.round(px).astype(np.int32)
            iy = np.round(py).astype(np.int32)

            valid = (ix >= 1) & (ix < rw - 1) & (iy >= 1) & (iy < rh - 1)
            if np.count_nonzero(valid) < 5:
                continue

            ix_v = ix[valid]
            iy_v = iy[valid]
            r_v = r_vals[valid]

            gx_v = gx[iy_v, ix_v]
            gy_v = gy[iy_v, ix_v]

            proj_n = gx_v * nx + gy_v * ny
            proj_t = np.abs(gx_v * tx + gy_v * ty)

            angle_ok = (proj_n > 2.0) & (proj_t <= proj_n * 0.55)
            if not np.any(angle_ok):
                continue

            resp = np.where(angle_ok, proj_n, 0.0)
            spatial_w = np.exp(-0.5 * ((r_v - half_side) / sigma_r) ** 2)
            weighted_resp = resp * spatial_w

            idx = int(np.argmax(weighted_resp))
            if resp[idx] <= 2.0:
                continue

            r_best = float(r_v[idx])
            if 0 < idx < len(resp) - 1:
                v_prev, v_curr, v_next = float(resp[idx - 1]), float(resp[idx]), float(resp[idx + 1])
                denom = 2.0 * (v_prev - 2.0 * v_curr + v_next)
                if abs(denom) > 1e-4:
                    r_best += float(np.clip((v_prev - v_next) / denom, -0.5, 0.5))

            pt_x = cx + s * tx + r_best * nx
            pt_y = cy + s * ty + r_best * ny
            pts_side.append((pt_x, pt_y))
            w_side.append(float(weighted_resp[idx]))

        line = _fit_line_ransac_2d(pts_side, w_side, expected_dir=(tx, ty), max_angle_deg=18.0)
        edge_lines.append(line)
        edge_weights_sum.append(float(np.sum(w_side)))
        edge_pts_list.append(pts_side)

    if any(l is None for l in edge_lines):
        return {
            'success': False,
            'error': 'Failed to reliably detect all 4 indent edges'
        }

    # 6. Mutual stabilization
    for i, j in ((0, 2), (1, 3)):
        vxi, vyi, x0i, y0i = edge_lines[i]
        vxj, vyj, x0j, y0j = edge_lines[j]
        if abs(vxi * vxj + vyi * vyj) < np.cos(np.deg2rad(10.0)):
            if edge_weights_sum[i] >= edge_weights_sum[j] and len(edge_pts_list[j]) > 0:
                pts_arr = np.asarray(edge_pts_list[j], dtype=np.float32)
                proj = -pts_arr[:, 0] * vyi + pts_arr[:, 1] * vxi
                med_idx = int(np.argsort(proj)[len(proj) // 2])
                edge_lines[j] = (vxi, vyi, float(pts_arr[med_idx, 0]), float(pts_arr[med_idx, 1]))
            elif len(edge_pts_list[i]) > 0:
                pts_arr = np.asarray(edge_pts_list[i], dtype=np.float32)
                proj = -pts_arr[:, 0] * vyj + pts_arr[:, 1] * vxj
                med_idx = int(np.argsort(proj)[len(proj) // 2])
                edge_lines[i] = (vxj, vyj, float(pts_arr[med_idx, 0]), float(pts_arr[med_idx, 1]))

    # 7. Corner intersections
    def intersect(l1, l2):
        vx1, vy1, x01, y01 = l1
        vx2, vy2, x02, y02 = l2
        det = -vx1 * vy2 + vy1 * vx2
        if abs(det) < 1e-5:
            return None
        t1 = (-(x02 - x01) * vy2 + (y02 - y01) * vx2) / det
        return np.array([x01 + t1 * vx1, y01 + t1 * vy1], dtype=np.float32)

    raw_corners = []
    for k in range(4):
        pt = intersect(edge_lines[k], edge_lines[(k + 1) % 4])
        if pt is None:
            return {'success': False, 'error': 'Failed to find edge intersections'}
        raw_corners.append(pt + np.array([rx1, ry1], dtype=np.float32))

    c_mean = np.mean(raw_corners, axis=0)
    angles = [float(np.arctan2(p[1] - c_mean[1], p[0] - c_mean[0])) for p in raw_corners]
    order = np.argsort(angles)
    ordered = [raw_corners[idx] for idx in order]

    top_idx = int(np.argmin([p[0] + p[1] for p in ordered]))
    ordered = [ordered[(top_idx + k) % 4] for k in range(4)]

    g_tl, g_tr, g_br, g_bl = ordered[0], ordered[1], ordered[2], ordered[3]

    d1_px = float(np.linalg.norm(g_tl - g_br))
    d2_px = float(np.linalg.norm(g_tr - g_bl))
    d_avg_px = (d1_px + d2_px) / 2.0

    sides = [float(np.linalg.norm(ordered[k] - ordered[(k + 1) % 4])) for k in range(4)]
    if min(d1_px, d2_px) < 20.0 or max(d1_px, d2_px) / min(d1_px, d2_px) > 1.60:
        return {
            'success': False,
            'error': f'Distorted indent geometry (D1={d1_px:.1f}, D2={d2_px:.1f} px, ratio={max(d1_px, d2_px) / max(1e-3, min(d1_px, d2_px)):.2f})'
        }
    if min(sides) <= 0 or max(sides) / min(sides) > 1.65:
        return {
            'success': False,
            'error': f'Indent side lengths differ significantly (min={min(sides):.1f}, max={max(sides):.1f} px)'
        }

    all_x = [p[0] for p in ordered]
    all_y = [p[1] for p in ordered]
    pad_zoom = 35
    zx1 = max(0, int(min(all_x) - pad_zoom))
    zy1 = max(0, int(min(all_y) - pad_zoom))
    zx2 = min(w_img, int(max(all_x) + pad_zoom))
    zy2 = min(h_img, int(max(all_y) + pad_zoom))

    return {
        'success': True,
        'corners': {
            'TL': (float(g_tl[0]), float(g_tl[1])),
            'TR': (float(g_tr[0]), float(g_tr[1])),
            'BR': (float(g_br[0]), float(g_br[1])),
            'BL': (float(g_bl[0]), float(g_bl[1]))
        },
        'd1_px': d1_px,
        'd2_px': d2_px,
        'd_avg_px': d_avg_px,
        'roi_bbox': (zx1, zy1, max(1, zx2 - zx1), max(1, zy2 - zy1)),
        'error': None
    }


def save_calibration(scale_um_per_px, nominal_val=37.1, d_avg_px=224.98, unit="um",
                     load_gf=300.0, reference_hv=None, camera_index=None,
                     filepath=DEFAULT_CALIBRATION_FILE):
    """Saves calibration parameters to a JSON file, preserving existing camera_index if not provided."""
    existing = load_calibration(filepath) or {}
    cam_idx = camera_index if camera_index is not None else existing.get("camera_index", 1)
    ref_hv = float(reference_hv) if reference_hv is not None else existing.get("reference_hv")

    data = {
        "camera_index": int(cam_idx),
        "reference_hv": ref_hv,
        "nominal_diagonal_val": float(nominal_val),
        "unit": unit,
        "measured_diagonal_px": float(d_avg_px),
        "scale_um_per_px": float(scale_um_per_px),
        "scale_px_per_um": float(1.0 / scale_um_per_px) if scale_um_per_px > 0 else 0.0,
        "scale_mm_per_px": float(scale_um_per_px / 1000.0),
        "default_load_gf": float(load_gf)
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return data


def save_camera_index(camera_index, filepath=DEFAULT_CALIBRATION_FILE):
    """
    Saves preferred camera index to calibration JSON file,
    preserving all existing scale and hardness calibration parameters.
    """
    data = load_calibration(filepath) or {}
    data["camera_index"] = int(camera_index)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving camera index to {filepath}: {e}")
        return False


def load_calibration(filepath=DEFAULT_CALIBRATION_FILE):
    """Loads calibration parameters from a JSON file."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading calibration file {filepath}: {e}")
        return None


def annotate_image(image_bgr, detection_result, scale_um_per_px=None, load_gf=300.0,
                   active_corner=None, hover_corner=None, show_instructions=False):
    """
    Creates an informative graphical overlay:
    - Yellow contour around 4 facets
    - Diagonals TL-BR (red) and TR-BL (blue)
    - Corner markers with interactive highlight (hover / active dragging)
    - Dynamic sidebar on unoccupied side of frame
    - Zoom Inset x2.5 magnification view
    - Vickers hardness (HV) calculations and metrics
    - Interactive controls guide when show_instructions=True
    """
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    if not detection_result.get('success', False) and 'corners' not in detection_result:
        cv2.putText(
            vis,
            f"Indent not detected: {detection_result.get('error', 'Error')}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        return vis

    corners = detection_result['corners']
    tl = np.array(corners['TL'], dtype=np.float32)
    tr = np.array(corners['TR'], dtype=np.float32)
    br = np.array(corners['BR'], dtype=np.float32)
    bl = np.array(corners['BL'], dtype=np.float32)

    # 1. Yellow facet contour
    pts_poly = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [pts_poly], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 2. Diagonals (red TL-BR, blue TR-BL)
    cv2.line(vis, tuple(tl.astype(int)), tuple(br.astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(vis, tuple(tr.astype(int)), tuple(bl.astype(int)), (255, 120, 0), 2, cv2.LINE_AA)

    mid1 = ((tl + br) / 2.0).astype(int)
    mid2 = ((tr + bl) / 2.0).astype(int)
    cv2.putText(vis, "D1", (mid1[0] + 6, mid1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "D2", (mid2[0] + 6, mid2[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1, cv2.LINE_AA)

    # 3. Corner markers
    for name in ['TL', 'TR', 'BR', 'BL']:
        pt = corners[name]
        ix, iy = int(round(pt[0])), int(round(pt[1]))

        if name == active_corner:
            cv2.circle(vis, (ix, iy), 10, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(vis, (ix - 16, iy), (ix + 16, iy), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (ix, iy - 16), (ix, iy + 16), (0, 255, 255), 1, cv2.LINE_AA)
            lbl = f"{name} ({ix},{iy})"
            col = (0, 255, 255)
        elif name == hover_corner:
            cv2.circle(vis, (ix, iy), 9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 3, (0, 255, 0), -1, cv2.LINE_AA)
            lbl = f"{name} [Drag with LMB]"
            col = (0, 255, 255)
        else:
            cv2.circle(vis, (ix, iy), 6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 2, (255, 255, 255), -1, cv2.LINE_AA)
            lbl = name
            col = (0, 255, 0)

        t_sz, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = ix + 9
        ty_t = iy - 7
        cv2.rectangle(vis, (tx - 2, ty_t - t_sz[1] - 2), (tx + t_sz[0] + 2, ty_t + 2), (20, 20, 20), -1)
        cv2.putText(vis, lbl, (tx, ty_t), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 4. Zoom ROI
    pts = np.array([tl, tr, br, bl])
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    span = int(max(max_x - min_x, max_y - min_y) * 1.35)
    span = max(80, min(min(h, w) - 10, span))
    cen_x = int(round(np.mean(pts[:, 0])))
    cen_y = int(round(np.mean(pts[:, 1])))
    rx = max(0, min(w - span, cen_x - span // 2))
    ry = max(0, min(h - span, cen_y - span // 2))
    rw_box = span
    rh_box = span
    detection_result['roi_bbox'] = (rx, ry, rw_box, rh_box)

    roi_marked = vis[ry:ry+rh_box, rx:rx+rw_box].copy()

    # 5. Position sidebar panel
    cx = float(cen_x)
    panel_w = 380
    panel_h = min(h - 24, 610)

    if cx < w / 2:
        px1 = w - panel_w - 15
    else:
        px1 = 15
    py1 = 12
    px2 = px1 + panel_w
    py2 = py1 + panel_h

    overlay = vis.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)
    cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 2)

    cv2.rectangle(vis, (px1, py1), (px2, py1 + 42), (35, 35, 35), -1)
    cv2.putText(vis, "VICKERS ANALYSIS RESULT", (px1 + 25, py1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    if detection_result.get('manual_adjusted', False):
        cv2.rectangle(vis, (px1 + 15, py1 + 46), (px2 - 15, py1 + 68), (0, 140, 255), -1)
        cv2.putText(vis, "MANUAL ADJUSTMENT ACTIVE", (px1 + 65, py1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        zoom_top = py1 + 75
    else:
        zoom_top = py1 + 52

    # 6. Zoom Inset x2.5
    zoom_size = 190
    if roi_marked.size > 0:
        zoom_img = cv2.resize(roi_marked, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
        zx1 = px1 + (panel_w - zoom_size) // 2
        zy1 = zoom_top
        zx2 = zx1 + zoom_size
        zy2 = zy1 + zoom_size

        cv2.rectangle(vis, (zx1 - 2, zy1 - 2), (zx2 + 2, zy2 + 2), (0, 200, 200), 1)
        vis[zy1:zy2, zx1:zx2] = zoom_img
        cv2.putText(vis, "Indent (Zoom x2.5)", (zx1 + 24, zy2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        detection_result['_zoom_coords'] = (zx1, zy1, zx2, zy2, rx, ry, rw_box, rh_box)

    # 7. Measurements and Vickers microhardness
    d1_px = float(detection_result['d1_px'])
    d2_px = float(detection_result['d2_px'])
    d_avg_px = float(detection_result['d_avg_px'])

    ty = zy2 + 34

    if scale_um_per_px:
        d1_um = d1_px * scale_um_per_px
        d2_um = d2_px * scale_um_per_px
        d_avg_um = d_avg_px * scale_um_per_px
        hv = calculate_vickers_hardness(d_avg_um, load_gf)

        cv2.rectangle(vis, (px1 + 15, ty), (px2 - 15, ty + 68), (25, 45, 25), -1)
        cv2.rectangle(vis, (px1 + 15, ty), (px2 - 15, ty + 68), (0, 255, 0), 1)
        cv2.putText(vis, f"HARDNESS (HV {load_gf/1000.0:.1f}):", (px1 + 25, ty + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{hv:.1f} HV", (px1 + 75, ty + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2, cv2.LINE_AA)

        ty += 88
        rows = [
            ("Load:", f"{int(load_gf)} g ({load_gf/1000.0:.3f} kgf)"),
            ("Diagonal D1:", f"{d1_um:.2f} um  ({d1_px:.1f} px)"),
            ("Diagonal D2:", f"{d2_um:.2f} um  ({d2_px:.1f} px)"),
            ("Mean D_avg:", f"{d_avg_um:.2f} um  ({d_avg_px:.1f} px)"),
            ("Scale:", f"{scale_um_per_px:.4f} um/px"),
        ]
    else:
        cv2.rectangle(vis, (px1 + 15, ty), (px2 - 15, ty + 50), (40, 40, 20), -1)
        cv2.putText(vis, "NOT CALIBRATED (Press C)", (px1 + 35, ty + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        ty += 70
        rows = [
            ("Load:", f"{int(load_gf)} g"),
            ("Diagonal D1:", f"{d1_px:.2f} px"),
            ("Diagonal D2:", f"{d2_px:.2f} px"),
            ("Mean D_avg:", f"{d_avg_px:.2f} px"),
        ]

    for label, val in rows:
        cv2.putText(vis, label, (px1 + 20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(vis, val, (px1 + 140, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        ty += 23

    # 8. Interactive controls guide
    if show_instructions and (ty + 60 <= py2 - 5):
        help_y = ty + 6
        cv2.rectangle(vis, (px1 + 15, help_y), (px2 - 15, help_y + 58), (28, 28, 28), -1)
        cv2.rectangle(vis, (px1 + 15, help_y), (px2 - 15, help_y + 58), (70, 70, 70), 1)
        cv2.putText(vis, "LMB Drag: Move corner points", (px1 + 25, help_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "[R]: Reset to auto-detection", (px1 + 25, help_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "SPACE/ENTER/ESC: Save & Exit", (px1 + 25, help_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 100), 1, cv2.LINE_AA)

    return vis


def annotate_calibration_overlay(image_bgr, corners, target_hv, load_gf=300.0,
                                 active_corner=None, hover_corner=None, show_instructions=True):
    """
    Creates specialized calibration overlay showing:
    - User-specified reference microhardness (HV)
    - Theoretical nominal diagonal (um) computed from target HV
    - Live measured diagonals in pixels
    - Resulting calculated optical scale (um/px and px/um)
    - Zoom Inset x2.5 with interactive point handles
    """
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    tl = np.array(corners['TL'], dtype=np.float32)
    tr = np.array(corners['TR'], dtype=np.float32)
    br = np.array(corners['BR'], dtype=np.float32)
    bl = np.array(corners['BL'], dtype=np.float32)

    # 1. Facet polygon
    pts_poly = np.array([tl, tr, br, bl], dtype=np.int32)
    cv2.polylines(vis, [pts_poly], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 2. Diagonals
    cv2.line(vis, tuple(tl.astype(int)), tuple(br.astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(vis, tuple(tr.astype(int)), tuple(bl.astype(int)), (255, 120, 0), 2, cv2.LINE_AA)

    mid1 = ((tl + br) / 2.0).astype(int)
    mid2 = ((tr + bl) / 2.0).astype(int)
    cv2.putText(vis, "D1", (mid1[0] + 6, mid1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "D2", (mid2[0] + 6, mid2[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 0), 1, cv2.LINE_AA)

    # 3. Corner markers
    for name in ['TL', 'TR', 'BR', 'BL']:
        pt = corners[name]
        ix, iy = int(round(pt[0])), int(round(pt[1]))
        if name == active_corner:
            cv2.circle(vis, (ix, iy), 10, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(vis, (ix - 16, iy), (ix + 16, iy), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (ix, iy - 16), (ix, iy + 16), (0, 255, 255), 1, cv2.LINE_AA)
            lbl = f"{name} ({ix},{iy})"
            col = (0, 255, 255)
        elif name == hover_corner:
            cv2.circle(vis, (ix, iy), 9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 3, (0, 255, 0), -1, cv2.LINE_AA)
            lbl = f"{name} [Drag with LMB]"
            col = (0, 255, 255)
        else:
            cv2.circle(vis, (ix, iy), 6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 2, (255, 255, 255), -1, cv2.LINE_AA)
            lbl = name
            col = (0, 255, 0)

        t_sz, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = ix + 9
        ty_t = iy - 7
        cv2.rectangle(vis, (tx - 2, ty_t - t_sz[1] - 2), (tx + t_sz[0] + 2, ty_t + 2), (20, 20, 20), -1)
        cv2.putText(vis, lbl, (tx, ty_t), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 4. Zoom ROI
    pts = np.array([tl, tr, br, bl])
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    span = int(max(max_x - min_x, max_y - min_y) * 1.35)
    span = max(80, min(min(h, w) - 10, span))
    cen_x, cen_y = int(round(np.mean(pts[:, 0]))), int(round(np.mean(pts[:, 1])))
    rx = max(0, min(w - span, cen_x - span // 2))
    ry = max(0, min(h - span, cen_y - span // 2))
    rw_box, rh_box = span, span

    roi_marked = vis[ry:ry+rh_box, rx:rx+rw_box].copy()

    # 5. Position sidebar
    panel_w = 380
    panel_h = min(h - 24, 610)
    px1 = w - panel_w - 15 if cen_x < w / 2 else 15
    py1 = 12
    px2 = px1 + panel_w
    py2 = py1 + panel_h

    overlay = vis.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)
    cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 2)

    # Panel header
    cv2.rectangle(vis, (px1, py1), (px2, py1 + 42), (40, 35, 15), -1)
    cv2.putText(vis, "VICKERS CALIBRATION", (px1 + 45, py1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Zoom Inset
    zoom_size = 190
    zx1 = px1 + (panel_w - zoom_size) // 2
    zy1 = py1 + 52
    zx2 = zx1 + zoom_size
    zy2 = zy1 + zoom_size
    if roi_marked.size > 0:
        zoom_img = cv2.resize(roi_marked, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(vis, (zx1 - 2, zy1 - 2), (zx2 + 2, zy2 + 2), (0, 200, 200), 1)
        vis[zy1:zy2, zx1:zx2] = zoom_img
        cv2.putText(vis, "Indent (Zoom x2.5)", (zx1 + 24, zy2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    zoom_coords = (zx1, zy1, zx2, zy2, rx, ry, rw_box, rh_box)

    # 6. Calculations
    d1_px = max(1.0, float(np.linalg.norm(tl - br)))
    d2_px = max(1.0, float(np.linalg.norm(tr - bl)))
    d_avg_px = (d1_px + d2_px) / 2.0

    d_nom_um = calculate_diagonal_from_hardness(target_hv, load_gf)
    scale_um_per_px = d_nom_um / d_avg_px
    scale_px_per_um = 1.0 / scale_um_per_px if scale_um_per_px > 0 else 0.0

    ty = zy2 + 34
    # Resulting scale box
    cv2.rectangle(vis, (px1 + 15, ty), (px2 - 15, ty + 68), (35, 30, 15), -1)
    cv2.rectangle(vis, (px1 + 15, ty), (px2 - 15, ty + 68), (0, 255, 255), 1)
    cv2.putText(vis, "CALIBRATED SCALE:", (px1 + 25, ty + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"{scale_um_per_px:.6f} um/px", (px1 + 35, ty + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

    ty += 88
    rows = [
        ("Reference HV:", f"{target_hv:.1f} HV ({int(load_gf)} gf)"),
        ("Nominal d_nom:", f"{d_nom_um:.2f} um"),
        ("Measured d_avg:", f"{d_avg_px:.1f} px"),
        ("Diagonal D1:", f"{d1_px:.1f} px"),
        ("Diagonal D2:", f"{d2_px:.1f} px"),
        ("Optical Scale:", f"{scale_px_per_um:.2f} px/um"),
    ]

    for label, val in rows:
        cv2.putText(vis, label, (px1 + 20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(vis, val, (px1 + 145, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        ty += 22

    if show_instructions and (ty + 75 <= py2 - 5):
        help_y = ty + 6
        cv2.rectangle(vis, (px1 + 15, help_y), (px2 - 15, help_y + 70), (28, 28, 28), -1)
        cv2.rectangle(vis, (px1 + 15, help_y), (px2 - 15, help_y + 70), (70, 70, 70), 1)
        cv2.putText(vis, "LMB Drag: Move corner points", (px1 + 25, help_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "[T]: Edit Target Hardness (HV)", (px1 + 25, help_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "[R]: Reset to auto-detection", (px1 + 25, help_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "SPACE/ENTER: Save & Apply | ESC: Cancel", (px1 + 25, help_y + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100), 1, cv2.LINE_AA)

    return vis, zoom_coords, scale_um_per_px, d_nom_um, d_avg_px


def interactive_measure_indent(
    image_bgr,
    detection_result=None,
    scale_um_per_px=None,
    load_gf=300.0,
    win_title="Measurement Result - Vickers Hardness",
    save_path=None
):
    """
    Interactive indentation measurement window:
    - Allows user to drag diagonal corner points (TL, TR, BR, BL) with mouse
      both in the main frame and inside the Zoom Inset x2.5 window.
    - Instantly recalculates diagonal lengths and Vickers hardness during dragging.
    - Key 'R' resets points to original auto-detection position.
    - Saves the annotated image to save_path ONLY at the end when the window is closed
      (by pressing SPACE, ENTER, ESC, Q, S, or clicking the window close button).
    """
    h_img, w_img = image_bgr.shape[:2]

    if detection_result is not None and detection_result.get('success', False) and 'corners' in detection_result:
        initial_corners = {k: np.array(v, dtype=np.float32) for k, v in detection_result['corners'].items()}
        was_detected = True
    else:
        cx, cy = w_img / 2.0, h_img / 2.0
        size = 110.0
        initial_corners = {
            'TL': np.array([cx, cy - size], dtype=np.float32),
            'TR': np.array([cx + size, cy], dtype=np.float32),
            'BR': np.array([cx, cy + size], dtype=np.float32),
            'BL': np.array([cx - size, cy], dtype=np.float32),
        }
        was_detected = False

    current_corners = {k: v.copy() for k, v in initial_corners.items()}

    class DragState:
        def __init__(self):
            self.dragged = None
            self.drag_mode = 'main'
            self.hover = None
            self.modified = not was_detected
            self.needs_redraw = True
            self.zoom_coords = None

    state = DragState()

    def on_mouse(event, x, y, flags, param):
        hit_corner = None
        hit_mode = 'main'

        if state.zoom_coords is not None:
            zx1, zy1, zx2, zy2, rx, ry, rw_box, rh_box = state.zoom_coords
            if zx1 <= x <= zx2 and zy1 <= y <= zy2:
                scale_zoom = 190.0 / rw_box
                for name in ['TL', 'TR', 'BR', 'BL']:
                    pt = current_corners[name]
                    sx = zx1 + (pt[0] - rx) * scale_zoom
                    sy = zy1 + (pt[1] - ry) * scale_zoom
                    if np.hypot(x - sx, y - sy) <= 15:
                        hit_corner = name
                        hit_mode = 'zoom'
                        break

        if hit_corner is None:
            for name in ['TL', 'TR', 'BR', 'BL']:
                pt = current_corners[name]
                if np.hypot(x - pt[0], y - pt[1]) <= 20:
                    hit_corner = name
                    hit_mode = 'main'
                    break

        if event == cv2.EVENT_LBUTTONDOWN:
            if hit_corner is not None:
                state.dragged = hit_corner
                state.drag_mode = hit_mode
                state.needs_redraw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.dragged is not None:
                corner_name = state.dragged
                if state.drag_mode == 'zoom' and state.zoom_coords is not None:
                    zx1, zy1, zx2, zy2, rx, ry, rw_box, rh_box = state.zoom_coords
                    rel_x = (x - zx1) / 190.0
                    rel_y = (y - zy1) / 190.0
                    new_x = np.clip(rx + rel_x * rw_box, 0, w_img - 1)
                    new_y = np.clip(ry + rel_y * rh_box, 0, h_img - 1)
                else:
                    new_x = np.clip(x, 0, w_img - 1)
                    new_y = np.clip(y, 0, h_img - 1)

                current_corners[corner_name] = np.array([new_x, new_y], dtype=np.float32)
                state.modified = True
                state.needs_redraw = True
            else:
                if state.hover != hit_corner:
                    state.hover = hit_corner
                    state.needs_redraw = True

        elif event == cv2.EVENT_LBUTTONUP:
            if state.dragged is not None:
                state.dragged = None
                state.needs_redraw = True

    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_title, on_mouse)

    last_res = None
    final_annotated = None

    try:
        while True:
            if state.needs_redraw:
                tl = current_corners['TL']
                tr = current_corners['TR']
                br = current_corners['BR']
                bl = current_corners['BL']

                d1_px = max(1.0, float(np.linalg.norm(tl - br)))
                d2_px = max(1.0, float(np.linalg.norm(tr - bl)))
                d_avg_px = (d1_px + d2_px) / 2.0

                last_res = {
                    'success': True,
                    'corners': {k: v.copy() for k, v in current_corners.items()},
                    'd1_px': d1_px,
                    'd2_px': d2_px,
                    'd_avg_px': d_avg_px,
                    'manual_adjusted': state.modified,
                }

                annotated = annotate_image(
                    image_bgr,
                    last_res,
                    scale_um_per_px=scale_um_per_px,
                    load_gf=load_gf,
                    active_corner=state.dragged,
                    hover_corner=state.hover,
                    show_instructions=True
                )
                state.zoom_coords = last_res.get('_zoom_coords')
                cv2.imshow(win_title, annotated)
                state.needs_redraw = False

            key = cv2.waitKey(20) & 0xFF
            if key in (32, 13, 27, ord('q'), ord('Q'), ord('s'), ord('S'), 233, 251):
                break
            elif key in (ord('r'), ord('R'), 234, 202):
                for k, v in initial_corners.items():
                    current_corners[k] = v.copy()
                state.modified = not was_detected
                state.needs_redraw = True

            try:
                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

    finally:
        if last_res is not None:
            final_annotated = annotate_image(
                image_bgr,
                last_res,
                scale_um_per_px=scale_um_per_px,
                load_gf=load_gf,
                active_corner=None,
                hover_corner=None,
                show_instructions=False
            )
            if save_path:
                try:
                    cv2.imwrite(str(save_path), final_annotated)
                    print(f"  [Save]: Final measurement frame saved to {Path(save_path).name}")
                except Exception as e:
                    print(f"  Error saving {save_path}: {e}")

        try:
            cv2.destroyWindow(win_title)
        except Exception:
            pass

    return last_res, final_annotated


def interactive_calibrate_indent(
    image_bgr,
    detection_result=None,
    target_hv=500.0,
    load_gf=300.0,
    win_title="Vickers Calibration - Reference Hardness Mode",
    save_path=None
):
    """
    Interactive calibration window:
    - User aligns corner points (TL, TR, BR, BL) to indent vertices on calibration test block.
    - User specifies certified reference microhardness (HV).
    - Theoretical nominal diagonal is computed:
        d_nom_um = sqrt(1.85437 * F_kgf / target_hv) * 1000.0
    - Live scale is computed:
        scale_um_per_px = d_nom_um / d_avg_px
    - Press [T] to edit target hardness anytime.
    - Press [SPACE] or [ENTER] to save calibration and update scale.
    - Press [ESC] to cancel calibration without saving.
    """
    h_img, w_img = image_bgr.shape[:2]

    if detection_result is not None and detection_result.get('success', False) and 'corners' in detection_result:
        initial_corners = {k: np.array(v, dtype=np.float32) for k, v in detection_result['corners'].items()}
        was_detected = True
    else:
        cx, cy = w_img / 2.0, h_img / 2.0
        size = 110.0
        initial_corners = {
            'TL': np.array([cx, cy - size], dtype=np.float32),
            'TR': np.array([cx + size, cy], dtype=np.float32),
            'BR': np.array([cx, cy + size], dtype=np.float32),
            'BL': np.array([cx - size, cy], dtype=np.float32),
        }
        was_detected = False

    current_corners = {k: v.copy() for k, v in initial_corners.items()}

    class CalibState:
        def __init__(self, hv):
            self.dragged = None
            self.drag_mode = 'main'
            self.hover = None
            self.modified = not was_detected
            self.needs_redraw = True
            self.zoom_coords = None
            self.target_hv = float(hv)
            self.confirmed = False

    state = CalibState(target_hv)

    def on_mouse(event, x, y, flags, param):
        hit_corner = None
        hit_mode = 'main'

        if state.zoom_coords is not None:
            zx1, zy1, zx2, zy2, rx, ry, rw_box, rh_box = state.zoom_coords
            if zx1 <= x <= zx2 and zy1 <= y <= zy2:
                scale_zoom = 190.0 / rw_box
                for name in ['TL', 'TR', 'BR', 'BL']:
                    pt = current_corners[name]
                    sx = zx1 + (pt[0] - rx) * scale_zoom
                    sy = zy1 + (pt[1] - ry) * scale_zoom
                    if np.hypot(x - sx, y - sy) <= 15:
                        hit_corner = name
                        hit_mode = 'zoom'
                        break

        if hit_corner is None:
            for name in ['TL', 'TR', 'BR', 'BL']:
                pt = current_corners[name]
                if np.hypot(x - pt[0], y - pt[1]) <= 20:
                    hit_corner = name
                    hit_mode = 'main'
                    break

        if event == cv2.EVENT_LBUTTONDOWN:
            if hit_corner is not None:
                state.dragged = hit_corner
                state.drag_mode = hit_mode
                state.needs_redraw = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.dragged is not None:
                corner_name = state.dragged
                if state.drag_mode == 'zoom' and state.zoom_coords is not None:
                    zx1, zy1, zx2, zy2, rx, ry, rw_box, rh_box = state.zoom_coords
                    rel_x = (x - zx1) / 190.0
                    rel_y = (y - zy1) / 190.0
                    new_x = np.clip(rx + rel_x * rw_box, 0, w_img - 1)
                    new_y = np.clip(ry + rel_y * rh_box, 0, h_img - 1)
                else:
                    new_x = np.clip(x, 0, w_img - 1)
                    new_y = np.clip(y, 0, h_img - 1)

                current_corners[corner_name] = np.array([new_x, new_y], dtype=np.float32)
                state.modified = True
                state.needs_redraw = True
            else:
                if state.hover != hit_corner:
                    state.hover = hit_corner
                    state.needs_redraw = True

        elif event == cv2.EVENT_LBUTTONUP:
            if state.dragged is not None:
                state.dragged = None
                state.needs_redraw = True

    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_title, on_mouse)

    last_res = None
    final_scale = None
    final_annotated = None

    try:
        while True:
            if state.needs_redraw:
                tl = current_corners['TL']
                tr = current_corners['TR']
                br = current_corners['BR']
                bl = current_corners['BL']

                d1_px = max(1.0, float(np.linalg.norm(tl - br)))
                d2_px = max(1.0, float(np.linalg.norm(tr - bl)))
                d_avg_px = (d1_px + d2_px) / 2.0

                last_res = {
                    'success': True,
                    'corners': {k: v.copy() for k, v in current_corners.items()},
                    'd1_px': d1_px,
                    'd2_px': d2_px,
                    'd_avg_px': d_avg_px,
                    'target_hv': state.target_hv,
                    'manual_adjusted': state.modified,
                }

                vis, zoom_coords, scale_val, d_nom_val, d_px_val = annotate_calibration_overlay(
                    image_bgr,
                    corners=current_corners,
                    target_hv=state.target_hv,
                    load_gf=load_gf,
                    active_corner=state.dragged,
                    hover_corner=state.hover,
                    show_instructions=True
                )
                state.zoom_coords = zoom_coords
                final_scale = scale_val
                cv2.imshow(win_title, vis)
                state.needs_redraw = False

            key = cv2.waitKey(20) & 0xFF

            # Confirm and save: ENTER, SPACE, S
            if key in (32, 13, ord('s'), ord('S'), 251):
                state.confirmed = True
                break
            # Cancel: ESC, Q
            elif key in (27, ord('q'), ord('Q'), 233):
                state.confirmed = False
                break
            # Edit target hardness: T (English or Russian 'Е')
            elif key in (ord('t'), ord('T'), 229, 197):
                new_hv = prompt_target_hardness(initial_val=state.target_hv, load_gf=load_gf)
                if new_hv is not None and new_hv > 0:
                    state.target_hv = float(new_hv)
                    state.needs_redraw = True
            # Reset points: R
            elif key in (ord('r'), ord('R'), 234, 202):
                for k, v in initial_corners.items():
                    current_corners[k] = v.copy()
                state.modified = not was_detected
                state.needs_redraw = True

            try:
                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    state.confirmed = True
                    break
            except Exception:
                break

    finally:
        if state.confirmed and last_res is not None and final_scale is not None:
            d_nom_um = calculate_diagonal_from_hardness(state.target_hv, load_gf)
            final_annotated, _, _, _, _ = annotate_calibration_overlay(
                image_bgr,
                corners=current_corners,
                target_hv=state.target_hv,
                load_gf=load_gf,
                active_corner=None,
                hover_corner=None,
                show_instructions=False
            )
            save_calibration(
                scale_um_per_px=final_scale,
                nominal_val=d_nom_um,
                d_avg_px=last_res['d_avg_px'],
                unit="um",
                load_gf=load_gf,
                reference_hv=state.target_hv
            )
            if save_path:
                try:
                    cv2.imwrite(str(save_path), final_annotated)
                    print(f"  [Save]: Calibration report image saved to {Path(save_path).name}")
                except Exception as e:
                    print(f"  Error saving {save_path}: {e}")
        else:
            final_scale = None

        try:
            cv2.destroyWindow(win_title)
        except Exception:
            pass

    return final_scale, last_res, final_annotated


def calibrate_from_image(image_path, target_hv=500.0, load_gf=300.0):
    """Calibrates scale from a reference image using user-specified hardness (HV)."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error opening image file: {image_path}")
        return None

    res = detect_indentation(img)
    if not res['success']:
        print(f"Calibration error: {res['error']}")
        return None

    d_avg_px = res['d_avg_px']
    d_nom_um = calculate_diagonal_from_hardness(target_hv, load_gf)
    scale_um_per_px = d_nom_um / d_avg_px
    calib = save_calibration(scale_um_per_px, d_nom_um, d_avg_px, unit="um", load_gf=load_gf, reference_hv=target_hv)

    print("=" * 65)
    print("CALIBRATION COMPLETED SUCCESSFULLY")
    print(f"Reference hardness: {target_hv:.1f} HV ({int(load_gf)} gf)")
    print(f"Theoretical diagonal: {d_nom_um:.2f} um")
    print(f"Measured diagonal: {d_avg_px:.2f} px")
    print(f"Scale: {scale_um_per_px:.6f} um/pixel ({calib['scale_px_per_um']:.4f} px/um)")
    print(f"Saved to: {DEFAULT_CALIBRATION_FILE}")
    print("=" * 65)

    out_path = Path(image_path).resolve().parent / f"{Path(image_path).stem}_calibrated.jpg"
    final_vis, _, _, _, _ = annotate_calibration_overlay(
        img,
        corners=res['corners'],
        target_hv=target_hv,
        load_gf=load_gf,
        show_instructions=False
    )
    cv2.imwrite(str(out_path), final_vis)
    print(f"Annotated calibration image saved: {out_path}")
    return calib


if __name__ == "__main__":
    hardness_dir = Path(__file__).resolve().parent

    if "--calibrate" in sys.argv:
        ref_hv = 500.0
        fpath = None
        args = [a for a in sys.argv[1:] if a != "--calibrate"]
        if len(args) >= 1:
            try:
                ref_hv = float(args[0])
            except ValueError:
                fpath = Path(args[0])
        if len(args) >= 2:
            fpath = Path(args[1])
        if fpath is None:
            jpgs = [p for p in hardness_dir.glob("frame_*.jpg")
                    if not p.stem.endswith("_measured") and not p.stem.endswith("_calibrated")]
            fpath = jpgs[0] if jpgs else None
        if fpath:
            calibrate_from_image(fpath, target_hv=ref_hv, load_gf=300.0)
    else:
        target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
        if target is None:
            jpgs = [p for p in hardness_dir.glob("frame_*.jpg")
                    if not p.stem.endswith("_measured") and not p.stem.endswith("_calibrated")]
            target = sorted(jpgs)[-1] if jpgs else None

        if target and target.exists():
            calib = load_calibration()
            scale = calib.get('scale_um_per_px', 0.164901) if calib else 0.164901
            load = calib.get('default_load_gf', 300.0) if calib else 300.0

            img = cv2.imread(str(target))
            if img is not None:
                res = detect_indentation(img)
                out_path = target.resolve().parent / f"{target.stem}_measured.jpg"

                if "--interactive" in sys.argv or "-i" in sys.argv:
                    print(f"Opening interactive window for {target.name}...")
                    interactive_measure_indent(img, res, scale, load, save_path=out_path)
                else:
                    if res['success']:
                        d_um = res['d_avg_px'] * scale
                        hv = calculate_vickers_hardness(d_um, load)
                        print(f"File: {target.name}")
                        print(f"D1: {res['d1_px']*scale:.2f} um, D2: {res['d2_px']*scale:.2f} um, D_avg: {d_um:.2f} um")
                        print(f"Hardness: {hv:.1f} HV {load/1000.0:.1f}")
                        annotated = annotate_image(img, res, scale, load)
                        cv2.imwrite(str(out_path), annotated)
                        print(f"Saved: {out_path}")
                    else:
                        print(f"Detection error: {res.get('error')}")
