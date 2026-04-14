"""
FACIAL EDITOR DIAGNOSTIC TOOL
Saves debug images showing exactly what each step does.
Run: python debug_facial.py path/to/image.jpg
Outputs go to ./debug_output/
"""

import cv2
import numpy as np
import os
import sys
import insightface

OUT = "./debug_output"
os.makedirs(OUT, exist_ok=True)

# ====================== INSIGHTFACE ======================
_fa = insightface.app.FaceAnalysis(name='buffalo_l',
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
_fa.prepare(ctx_id=-1, det_size=(640, 640))


def detect_landmarks(bgr):
    faces = _fa.get(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not faces:
        return None
    return max(faces,
               key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
               ).landmark_2d_106.astype(np.float32)


def face_geo(lm):
    jaw = lm[0:33, :2]
    xs, ys = jaw[:, 0], jaw[:, 1]
    return {"cx": xs.mean(), "cy": ys.mean(),
            "left": xs.min(), "right": xs.max(),
            "top": ys.min(), "bottom": ys.max(),
            "w": xs.max()-xs.min(), "h": ys.max()-ys.min()}


def save(name, img):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print(f"  saved → {path}")


# ====================== TEST 1: LANDMARK OVERLAY ======================
def test_landmarks(bgr, lm):
    print("\n[TEST 1] Landmark positions")
    vis = bgr.copy()
    g   = face_geo(lm)

    # Draw all 106 landmarks
    for i, (x, y) in enumerate(lm):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (int(x)+3, int(y)-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)

    brow_y = np.mean([lm[43:52, 1].mean(), lm[97:106, 1].mean()])
    brow_top_y = min(lm[43:52, 1].min(), lm[97:106, 1].min())
    eye_y  = np.mean([lm[35:42, 1].mean(), lm[87:94, 1].mean()])

    # Draw key horizontal lines
    cv2.line(vis, (0, int(brow_y)),     (bgr.shape[1], int(brow_y)),     (0, 0, 255),   1)  # red  = brow center
    cv2.line(vis, (0, int(brow_top_y)), (bgr.shape[1], int(brow_top_y)), (0, 165, 255), 1)  # orange = brow top
    cv2.line(vis, (0, int(eye_y)),      (bgr.shape[1], int(eye_y)),      (255, 0, 0),   1)  # blue = eye center

    # Estimated hairline
    hairline_geo = max(10, int(brow_y - g['w'] * 0.40))
    cv2.line(vis, (0, hairline_geo), (bgr.shape[1], hairline_geo), (0, 255, 255), 1)  # yellow = hairline estimate

    print(f"  brow_y        = {brow_y:.1f}  (red line)")
    print(f"  brow_top_y    = {brow_top_y:.1f}  (orange line)")
    print(f"  eye_y         = {eye_y:.1f}  (blue line)")
    print(f"  hairline_est  = {hairline_geo}  (yellow line)")
    print(f"  face_geo top  = {g['top']:.1f}  (jaw top — NOTE: often BELOW brow_y)")
    print(f"  face_geo w    = {g['w']:.1f}")
    print(f"  face_geo h    = {g['h']:.1f}")

    # Add legend
    legend = [
        ("red   = brow center Y",    (0,0,255)),
        ("orange= brow TOP Y",        (0,165,255)),
        ("blue  = eye center Y",      (255,0,0)),
        ("yellow= hairline estimate", (0,255,255)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(vis, txt, (10, 20+i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    save("01_landmarks.jpg", vis)


# ====================== TEST 2: HAIRLINE PIXEL SCAN ======================
def test_hairline_scan(bgr, lm):
    print("\n[TEST 2] Pixel-based hairline scan")
    g      = face_geo(lm)
    brow_y = int(np.mean([lm[43:52, 1].mean(), lm[97:106, 1].mean()]))
    h, w   = bgr.shape[:2]
    vis    = bgr.copy()

    x_l = int(g['left']  + g['w'] * 0.30)
    x_r = int(g['right'] - g['w'] * 0.30)

    # Skin sample zone
    skin_y1 = max(0, brow_y - int(g['w'] * 0.15))
    skin_y2 = max(0, brow_y - int(g['w'] * 0.08))
    cv2.rectangle(vis, (x_l, skin_y1), (x_r, skin_y2), (255, 0, 255), 2)

    skin_patch = bgr[skin_y1:skin_y2, x_l:x_r]
    if skin_patch.size == 0:
        print("  ERROR: skin patch empty!")
        return
    skin_color = skin_patch.mean(axis=(0, 1)).astype(np.float32)
    print(f"  skin color BGR = {skin_color}")

    # Scan columns
    scan_cols   = np.linspace(x_l, x_r, 12).astype(int)
    hairline_ys = []

    for col in scan_cols:
        col = int(np.clip(col, 0, w-1))
        found = None
        for row in range(0, brow_y):
            px   = bgr[row, col].astype(np.float32)
            dist = float(np.linalg.norm(px - skin_color))
            if dist < 60.0:
                found = row
                break
        if found is not None:
            hairline_ys.append(found)
            cv2.circle(vis, (col, found), 4, (0, 255, 0), -1)  # green dot = detected hairline point
        else:
            cv2.line(vis, (col, 0), (col, brow_y), (0, 0, 255), 1)  # red = no skin found

    if len(hairline_ys) >= 3:
        median_hl = int(np.median(hairline_ys))
        cv2.line(vis, (0, median_hl), (w, median_hl), (0, 255, 0), 2)  # green = final hairline
        print(f"  detected hairline_y = {median_hl}  (green line)")
        print(f"  individual scans    = {sorted(hairline_ys)}")
    else:
        fallback = max(10, brow_y - int(g['w'] * 0.38))
        cv2.line(vis, (0, fallback), (w, fallback), (0, 0, 255), 2)
        print(f"  FALLBACK hairline   = {fallback}  (red line — scan failed)")

    # Mark scan zone
    cv2.rectangle(vis, (x_l, 0), (x_r, brow_y), (200, 200, 0), 1)
    cv2.putText(vis, "scan zone", (x_l+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
    cv2.putText(vis, "skin sample", (x_l+2, skin_y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    save("02_hairline_scan.jpg", vis)


# ====================== TEST 3: LOCAL FOREHEAD WARP ZONES ======================
def _get_local_forehead_params(bgr, lm):
    """Replicates app_claude_v3 localized forehead params for debug visualization."""
    h, w = bgr.shape[:2]
    g = face_geo(lm)
    jaw = lm[0:33, :2]
    brow_y = float(np.mean([lm[43:52, 1].mean(), lm[97:106, 1].mean()]))

    x_l = int(g['left'] + g['w'] * 0.30)
    x_r = int(g['right'] - g['w'] * 0.30)
    if x_r <= x_l + 4:
        x_l = int(g['left'] + g['w'] * 0.20)
        x_r = int(g['right'] - g['w'] * 0.20)
    if x_r <= x_l + 4:
        x_l = int(g['left'])
        x_r = int(g['right'])

    skin_y1 = max(0, int(brow_y) - int(g['w'] * 0.15))
    skin_y2 = max(skin_y1 + 1, int(brow_y) - int(g['w'] * 0.08))
    skin_patch = bgr[skin_y1:skin_y2, max(0, x_l):min(w, x_r)]

    hairline_y = max(10, int(brow_y - g['w'] * 0.38))
    if skin_patch.size > 0:
        skin_color = skin_patch.mean(axis=(0, 1)).astype(np.float32)
        scan_cols = np.linspace(max(0, x_l), min(w - 1, x_r), 12).astype(int)
        hits = []
        for col in scan_cols:
            for row in range(0, int(brow_y)):
                px = bgr[row, col].astype(np.float32)
                if float(np.linalg.norm(px - skin_color)) < 60.0:
                    hits.append(row)
                    break
        if len(hits) >= 3:
            hairline_y = int(np.median(hits))

    forehead_h = max(brow_y - hairline_y, 12.0)
    y_top = max(0.0, hairline_y - max(forehead_h * 1.2, 18.0))
    y_bottom = min(float(h - 1), hairline_y + max(forehead_h * 0.18, 6.0))

    left = float(jaw[:, 0].min())
    right = float(jaw[:, 0].max())
    face_w = right - left
    x_left = max(0.0, left - face_w * 0.02)
    x_right = min(float(w - 1), right + face_w * 0.02)

    return {
        "brow_y": brow_y,
        "hairline_y": float(hairline_y),
        "forehead_h": float(forehead_h),
        "y_top": float(y_top),
        "y_bottom": float(y_bottom),
        "x_left": float(x_left),
        "x_right": float(x_right),
    }


def test_strip_warp_zones(bgr, lm):
    print("\n[TEST 3] Local forehead warp zones — where weights are non-zero")
    h, w = bgr.shape[:2]
    p = _get_local_forehead_params(bgr, lm)

    y_top = p["y_top"]
    y_bottom = p["y_bottom"]
    x_left = p["x_left"]
    x_right = p["x_right"]

    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)

    band_h = y_bottom - y_top
    feather_y = max(band_h * 0.20, 8.0)
    wy = np.zeros(h, dtype=np.float32)
    ramp_up_end = y_top + feather_y
    ramp_dn_start = y_bottom - feather_y
    m = (rows >= ramp_up_end) & (rows < ramp_dn_start)
    wy[m] = 1.0
    m = (rows >= y_top) & (rows < ramp_up_end)
    wy[m] = (rows[m] - y_top) / max(feather_y, 1e-6)
    m = (rows >= ramp_dn_start) & (rows < y_bottom)
    wy[m] = (y_bottom - rows[m]) / max(feather_y, 1e-6)

    wx = np.zeros(w, dtype=np.float32)
    span = x_right - x_left
    feather_x = max(span * 0.16, 10.0)
    full_l = x_left + feather_x
    full_r = x_right - feather_x
    m = (cols >= full_l) & (cols <= full_r)
    wx[m] = 1.0
    m = (cols >= x_left) & (cols < full_l)
    wx[m] = (cols[m] - x_left) / max(feather_x, 1e-6)
    m = (cols > full_r) & (cols <= x_right)
    wx[m] = (x_right - cols[m]) / max(feather_x, 1e-6)

    center = (x_left + x_right) * 0.5
    sigma_x = max(span * 0.38, 16.0)
    bell = np.exp(-((cols - center) ** 2) / (2.0 * sigma_x ** 2))
    wx *= (0.55 + 0.45 * bell)

    weight_2d = np.outer(wy, wx).astype(np.float32)

    heat = cv2.applyColorMap((weight_2d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis = cv2.addWeighted(bgr, 0.72, heat, 0.28, 0)

    cv2.rectangle(vis, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (255, 255, 255), 1)
    cv2.line(vis, (0, int(p["brow_y"])), (w, int(p["brow_y"])), (0, 0, 255), 1)
    cv2.line(vis, (0, int(p["hairline_y"])), (w, int(p["hairline_y"])), (0, 255, 255), 1)

    cv2.putText(vis, "white box = localized warp bounds", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(vis, "red=brow, yellow=hairline", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    print(f"  x range       = [{x_left:.1f}, {x_right:.1f}]")
    print(f"  y range       = [{y_top:.1f}, {y_bottom:.1f}]")
    print(f"  brow_y        = {p['brow_y']:.1f}")
    print(f"  hairline_y    = {p['hairline_y']:.1f}")
    print(f"  forehead_h    = {p['forehead_h']:.1f}")
    print("  expectation   = effect limited around forehead center, not full frame width")

    save("03_strip_warp_zones.jpg", vis)


# ====================== TEST 4: ACTUAL LOCAL WARP DIFF ======================
def apply_local_strip(img, y_top, y_bottom, dy, x_left, x_right):
    h, w = img.shape[:2]
    y_top = float(np.clip(y_top, 0, h - 1))
    y_bottom = float(np.clip(y_bottom, 0, h - 1))
    x_left = float(np.clip(x_left, 0, w - 1))
    x_right = float(np.clip(x_right, 0, w - 1))
    if y_bottom <= y_top or abs(dy) < 0.5 or x_right <= x_left + 2:
        return img

    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)

    band_h = y_bottom - y_top
    feather_y = max(band_h * 0.20, 8.0)
    wy = np.zeros(h, dtype=np.float32)
    ramp_up_end = y_top + feather_y
    ramp_dn_start = y_bottom - feather_y
    m = (rows >= ramp_up_end) & (rows < ramp_dn_start)
    wy[m] = 1.0
    m = (rows >= y_top) & (rows < ramp_up_end)
    wy[m] = (rows[m] - y_top) / max(feather_y, 1e-6)
    m = (rows >= ramp_dn_start) & (rows < y_bottom)
    wy[m] = (y_bottom - rows[m]) / max(feather_y, 1e-6)

    wx = np.zeros(w, dtype=np.float32)
    span = x_right - x_left
    feather_x = max(span * 0.16, 10.0)
    full_l = x_left + feather_x
    full_r = x_right - feather_x
    m = (cols >= full_l) & (cols <= full_r)
    wx[m] = 1.0
    m = (cols >= x_left) & (cols < full_l)
    wx[m] = (cols[m] - x_left) / max(feather_x, 1e-6)
    m = (cols > full_r) & (cols <= x_right)
    wx[m] = (x_right - cols[m]) / max(feather_x, 1e-6)

    center = (x_left + x_right) * 0.5
    sigma_x = max(span * 0.38, 16.0)
    bell = np.exp(-((cols - center) ** 2) / (2.0 * sigma_x ** 2))
    wx *= (0.55 + 0.45 * bell)

    mx = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
    my = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)
    weight_2d = np.outer(wy, wx).astype(np.float32)
    my -= dy * weight_2d

    return cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)


def test_actual_warp(bgr, lm):
    print("\n[TEST 4] Actual localized warp results at intensity 0.6 and 1.4")
    p = _get_local_forehead_params(bgr, lm)
    y_top, y_bottom = p["y_top"], p["y_bottom"]
    x_left, x_right = p["x_left"], p["x_right"]
    forehead_h = p["forehead_h"]

    for intensity, label in [(1.4, "max"), (0.6, "min")]:
        dy = -(intensity - 1.0) * forehead_h * 0.55
        print(f"  intensity={intensity}: dy={dy:.1f}px")
        result = apply_local_strip(bgr, y_top, y_bottom, dy, x_left, x_right)

        side = np.hstack([bgr, result])
        for offset in [0, bgr.shape[1]]:
            cv2.rectangle(
                side,
                (offset + int(x_left), int(y_top)),
                (offset + int(x_right), int(y_bottom)),
                (255, 255, 255),
                1,
            )
        cv2.putText(side, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            side,
            f"intensity={intensity} dy={dy:.0f}px",
            (bgr.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        save(f"04_warp_{label}.jpg", side)

    dy_max = -(1.4 - 1.0) * forehead_h * 0.55
    result_max = apply_local_strip(bgr, y_top, y_bottom, dy_max, x_left, x_right)
    diff = cv2.absdiff(bgr, result_max)
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)
    save("04_warp_diff_amplified.jpg", diff_amplified)
    print("  diff image saved (amplified 5x) — expected: localized bright region around forehead only")


# ====================== TEST 5: FACE MASK COVERAGE ======================
def test_face_mask(bgr, lm):
    print("\n[TEST 5] Face mask coverage")
    h, w   = bgr.shape[:2]
    jaw    = lm[0:33, :2].astype(np.int32)
    face_w = int(jaw[:, 0].max() - jaw[:, 0].min())
    cx     = int(jaw[:, 0].mean())
    brow_y = int(np.mean([lm[43:52, 1].mean(), lm[97:106, 1].mean()]))

    forehead_top = max(0, brow_y - int(face_w * 0.25))
    forehead_pts = np.array([
        [jaw[:,0].min() + int(face_w*0.10), brow_y],
        [cx - int(face_w*0.20),              forehead_top],
        [cx,                                 forehead_top],
        [cx + int(face_w*0.20),             forehead_top],
        [jaw[:,0].max() - int(face_w*0.10), brow_y],
    ], dtype=np.int32)

    full_poly = np.vstack([jaw, forehead_pts])
    mask      = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(full_poly), 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    mask   = cv2.GaussianBlur(mask, (15, 15), 0)

    # Overlay mask on image
    mask_f  = mask.astype(np.float32) / 255.0
    overlay = bgr.astype(np.float32).copy()
    overlay[:,:,1] = np.clip(overlay[:,:,1] + mask_f * 60, 0, 255)  # green tint on mask area

    # Draw hairline for comparison
    hairline_y = max(10, brow_y - int(face_w * 0.40))
    cv2.line(overlay.astype(np.uint8), (0, hairline_y), (w, hairline_y), (0,255,255), 2)
    cv2.putText(overlay.astype(np.uint8), "hairline", (10, hairline_y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(overlay.astype(np.uint8), "green=mask coverage", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

    save("05_face_mask.jpg", overlay.astype(np.uint8))

    # Check if hairline is inside or outside mask
    mask_at_hairline = mask[hairline_y, w//2]
    print(f"  mask value at hairline ({hairline_y}px): {mask_at_hairline}/255")
    print(f"  → {'INSIDE mask (forehead warp will be blended)' if mask_at_hairline > 50 else 'OUTSIDE mask (good — hair warp not affected by mask)'}")
    print(f"  forehead_top of mask = {forehead_top}px")
    print(f"  hairline estimate    = {hairline_y}px")
    print(f"  {'✅ hairline is above mask top — hair warp unaffected' if hairline_y < forehead_top else '⚠️  hairline is BELOW mask top — mask might interfere with hair warp'}")


# ====================== MAIN ======================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_facial.py path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    bgr      = cv2.imread(img_path)
    if bgr is None:
        print(f"ERROR: could not read {img_path}")
        sys.exit(1)

    print(f"Image: {img_path}  shape={bgr.shape}")

    lm = detect_landmarks(bgr)
    if lm is None:
        print("ERROR: No face detected")
        sys.exit(1)
    print(f"Face detected. Running tests...")

    test_landmarks(bgr, lm)
    test_hairline_scan(bgr, lm)
    test_strip_warp_zones(bgr, lm)
    test_actual_warp(bgr, lm)
    test_face_mask(bgr, lm)

    print(f"\n✅ All tests done. Check images in {OUT}/")
    print("Key files to check:")
    print("  01_landmarks.jpg         — see where all landmark lines are")
    print("  02_hairline_scan.jpg     — see if pixel scan finds real hairline")
    print("  03_strip_warp_zones.jpg  — see exactly which rows get moved")
    print("  04_warp_max.jpg          — side-by-side at intensity 1.4")
    print("  04_warp_min.jpg          — side-by-side at intensity 0.6")
    print("  04_warp_diff_amplified.jpg — white = changed pixels (5x amplified)")
    print("  05_face_mask.jpg         — green = mask area, check if it covers hairline")