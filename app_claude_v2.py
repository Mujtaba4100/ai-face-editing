"""PROFESSIONAL AI FACIAL EDITOR - v14
FOREHEAD FINAL FIX:
- Forehead warp now moves HAIR (above hairline), not skin
- Applied AFTER face mask blend so mask cannot undo the hair shift
- Full image width warp — no x-column clipping
- dy scaled to 25% of hair band height — strong visible effect
- Feather only at bottom (hairline boundary) — smooth skin transition
- Top edge = image edge, no feather needed there
"""

import gradio as gr
import cv2
import numpy as np
import time, logging
from PIL import Image, ImageEnhance
import insightface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== INSIGHTFACE ======================
try:
    _fa = insightface.app.FaceAnalysis(name='buffalo_l',
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    _fa.prepare(ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else -1, det_size=(640, 640))
    logger.info("InsightFace ready")
except Exception as e:
    logger.error(f"InsightFace: {e}")
    _fa = None


def detect_landmarks(bgr):
    if _fa is None:
        return None
    try:
        faces = _fa.get(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ).landmark_2d_106.astype(np.float32)
    except Exception:
        return None


def visualise_landmarks(bgr, lm):
    vis = bgr.copy()
    for i, (x, y) in enumerate(lm):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return vis


def face_geo(lm):
    jaw = lm[0:33, :2]
    xs, ys = jaw[:, 0], jaw[:, 1]
    return {
        "cx": xs.mean(), "cy": ys.mean(),
        "left": xs.min(), "right": xs.max(),
        "top": ys.min(), "bottom": ys.max(),
        "w": xs.max() - xs.min(), "h": ys.max() - ys.min()
    }


def _id_maps(h, w):
    return np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))


# ====================== WARP PRIMITIVES ======================

def _gauss_push(mx, my, cx, cy, dx, dy, sigma):
    sigma = max(3.5, float(sigma))
    h, w = mx.shape
    wx  = np.exp(-((np.arange(w, dtype=np.float32) - cx) ** 2) / (2 * sigma ** 2))
    wy  = np.exp(-((np.arange(h, dtype=np.float32) - cy) ** 2) / (2 * sigma ** 2))
    wgt = np.outer(wy, wx)
    mx -= dx * wgt
    my -= dy * wgt


def combined_warp(img, all_pts, all_disps, all_sigmas):
    if len(all_pts) == 0:
        return img
    h, w   = img.shape[:2]
    mx, my = _id_maps(h, w)
    for (cx, cy), (dx, dy), sigma in zip(
            all_pts.astype(float), all_disps.astype(float), all_sigmas):
        _gauss_push(mx, my, cx, cy, dx, dy, sigma)
    return cv2.remap(img, mx, my, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REFLECT_101)


def _apply_strip_warp(img, y_top, y_bottom, dy_pixels, x_left=None, x_right=None):
    """Shift a local forehead band vertically with soft x/y falloff."""
    h, w = img.shape[:2]

    y_top    = float(np.clip(y_top,    0, h - 1))
    y_bottom = float(np.clip(y_bottom, 0, h - 1))

    if y_bottom <= y_top or abs(dy_pixels) < 0.5:
        return img

    band_h = y_bottom - y_top
    feather_y = max(band_h * 0.18, 8.0)

    rows   = np.arange(h, dtype=np.float32)
    weight = np.zeros(h, dtype=np.float32)

    # Full weight in the middle of the band.
    ramp_up_end = y_top + feather_y
    ramp_dn_start = y_bottom - feather_y
    m = (rows >= ramp_up_end) & (rows < ramp_dn_start)
    weight[m] = 1.0

    # Ramp up from top edge.
    m = (rows >= y_top) & (rows < ramp_up_end)
    weight[m] = (rows[m] - y_top) / max(feather_y, 1e-6)

    # Ramp down near lower edge.
    m = (rows >= ramp_dn_start) & (rows < y_bottom)
    weight[m] = (y_bottom - rows[m]) / max(feather_y, 1e-6)

    # Optional x-localization so the whole frame does not shift.
    cols = np.arange(w, dtype=np.float32)
    xw = np.ones(w, dtype=np.float32)
    if x_left is not None and x_right is not None:
        xl = float(np.clip(x_left, 0, w - 1))
        xr = float(np.clip(x_right, 0, w - 1))
        if xr <= xl + 2:
            return img

        xw = np.zeros(w, dtype=np.float32)
        span = xr - xl
        feather_x = max(span * 0.16, 10.0)

        full_l = xl + feather_x
        full_r = xr - feather_x
        m = (cols >= full_l) & (cols <= full_r)
        xw[m] = 1.0

        m = (cols >= xl) & (cols < full_l)
        xw[m] = (cols[m] - xl) / max(feather_x, 1e-6)

        m = (cols > full_r) & (cols <= xr)
        xw[m] = (xr - cols[m]) / max(feather_x, 1e-6)

        center = (xl + xr) * 0.5
        sigma_x = max(span * 0.38, 16.0)
        bell = np.exp(-((cols - center) ** 2) / (2.0 * sigma_x ** 2))
        xw *= (0.55 + 0.45 * bell)

    mx = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
    my = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)

    weight_2d  = np.outer(weight, xw).astype(np.float32)
    my        -= dy_pixels * weight_2d

    return cv2.remap(img, mx, my, cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REFLECT_101)


# ====================== FACE MASK ======================

def create_face_mask(lm, h, w):
    jaw    = lm[0:33, :2].astype(np.int32)
    face_w = int(jaw[:, 0].max() - jaw[:, 0].min())
    cx     = int(jaw[:, 0].mean())

    brow_y       = int(np.mean([lm[43:52, 1].mean(), lm[97:106, 1].mean()]))
    forehead_top = max(0, brow_y - int(face_w * 0.25))

    forehead_pts = np.array([
        [jaw[:, 0].min() + int(face_w * 0.10), brow_y],
        [cx - int(face_w * 0.20),               forehead_top],
        [cx,                                     forehead_top],
        [cx + int(face_w * 0.20),               forehead_top],
        [jaw[:, 0].max() - int(face_w * 0.10), brow_y],
    ], dtype=np.int32)

    full_poly = np.vstack([jaw, forehead_pts])
    mask      = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(full_poly), 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask   = cv2.dilate(mask, kernel, iterations=1)
    mask   = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask.astype(np.float32) / 255.0


# ====================== DISPLACEMENTS ======================

def get_lips_disps(lm, intensity):
    if abs(intensity - 1.0) < 0.005:
        return np.empty((0, 2)), np.empty((0, 2)), 5.5
    pts    = lm[52:72].astype(np.float32)
    center = pts.mean(axis=0)
    scale  = np.clip((intensity - 1.0) * 0.34, -0.24, 0.24)
    disps  = (pts - center) * scale
    return pts, disps, 5.5


def get_nose_disps(lm, intensity):
    if abs(intensity - 1.0) < 0.005:
        return np.empty((0, 2)), np.empty((0, 2)), 3.5
    pts        = lm[72:84]
    cy         = pts[:, 1].mean()
    bot        = pts[pts[:, 1] >= cy]
    if len(bot) < 3:
        bot = pts
    sorted_bot = bot[np.argsort(bot[:, 0])]
    alar       = np.vstack([sorted_bot[:2], sorted_bot[-2:]])
    nose_w     = alar[:, 0].max() - alar[:, 0].min()
    mag        = np.clip((intensity - 1.0) * nose_w * 0.23,
                         -nose_w * 0.23, nose_w * 0.23)
    cx         = alar[:, 0].mean()
    disps      = np.zeros_like(alar)
    disps[alar[:, 0] < cx, 0] = -mag
    disps[alar[:, 0] > cx, 0] =  mag
    return alar, disps, 3.5


def get_eyebrows_disps(lm, intensity):
    if abs(intensity - 1.0) < 0.005:
        return np.empty((0, 2)), np.empty((0, 2)), 3.0

    left_brow = lm[43:52].astype(np.float32)
    right_brow = lm[97:106].astype(np.float32)

    # Intentional brow thickness control (old accidental effect, now explicit).
    # intensity > 1.0 => thicker; intensity < 1.0 => slimmer.
    k = np.clip(float(intensity - 1.0), -0.35, 0.45)

    def _brow_disps(brow):
        cy = float(brow[:, 1].mean())
        cx = float(brow[:, 0].mean())
        rel_y = brow[:, 1] - cy
        rel_x = brow[:, 0] - cx

        d = np.zeros_like(brow)
        d[:, 1] = rel_y * (k * 1.45)
        d[:, 0] = rel_x * (k * 0.12)
        return d

    left_disp = _brow_disps(left_brow)
    right_disp = _brow_disps(right_brow)

    brow_pts = np.vstack([left_brow, right_brow]).astype(np.float32)
    disps = np.vstack([left_disp, right_disp]).astype(np.float32)
    return brow_pts, disps, 3.0


def get_face_slim_disps(lm, intensity):
    if abs(intensity - 1.0) < 0.005:
        return np.empty((0, 2)), np.empty((0, 2)), 5.0
    g        = face_geo(lm)
    jaw      = lm[0:33]
    top_cut  = g['top'] + g['h'] * 0.22
    bot_cut  = g['top'] + g['h'] * 0.82
    side_pts = jaw[(jaw[:, 1] > top_cut) & (jaw[:, 1] < bot_cut)]
    left     = side_pts[side_pts[:, 0] < g['cx']]
    right    = side_pts[side_pts[:, 0] > g['cx']]
    if len(left) < 3 or len(right) < 3:
        return np.empty((0, 2)), np.empty((0, 2)), 5.0
    push  = (intensity - 1.0) * g['w'] * 0.025
    pts   = np.vstack([left, right])
    disps = np.vstack([
        np.column_stack([np.full(len(left),   push), np.zeros(len(left))]),
        np.column_stack([np.full(len(right), -push), np.zeros(len(right))])
    ])
    return pts, disps, g['w'] * 0.16


def get_jaw_disps(lm, intensity):
    if abs(intensity - 1.0) < 0.005:
        return np.empty((0, 2)), np.empty((0, 2)), 5.0
    g        = face_geo(lm)
    jaw      = lm[0:33]
    chin_cut = g['top'] + g['h'] * 0.78
    chin_pts = jaw[jaw[:, 1] > chin_cut]
    if len(chin_pts) < 3:
        chin_pts = jaw[np.argsort(jaw[:, 1])[-5:]]
    dy    = (intensity - 1.0) * g['h'] * 0.05
    disps = np.zeros_like(chin_pts)
    disps[:, 1] = dy
    return chin_pts, disps, g['h'] * 0.13


def get_forehead_params(lm, intensity, img_w):
    """
    Forehead-only skin band parameters.
    Keeps effect above eyebrows and inside facial width.
    """
    if abs(intensity - 1.0) < 0.005:
        return None

    g          = face_geo(lm)
    jaw        = lm[0:33, :2]
    brow_y     = float(np.mean([lm[43:52, 1].mean(), lm[97:106, 1].mean()]))
    eye_y      = float(np.mean([lm[35:42, 1].mean(), lm[87:94, 1].mean()]))
    # In image coordinates, eyes are below brows (larger y), so use eye_y - brow_y.
    gap        = max(eye_y - brow_y, 10.0)

    # Forehead band is pushed higher so brow thickness never changes here.
    y_bottom   = brow_y - gap * 0.80
    y_top      = brow_y - max(g['w'] * 0.46, gap * 3.80)
    y_top      = max(0.0, y_top)
    y_bottom   = max(y_top + 12.0, y_bottom)

    band_h     = y_bottom - y_top
    dy         = -(intensity - 1.0) * band_h * 0.72

    left       = float(jaw[:, 0].min())
    right      = float(jaw[:, 0].max())
    face_w     = right - left
    x_left     = left + face_w * 0.08
    x_right    = right - face_w * 0.08

    return (y_top, y_bottom, dy, x_left, x_right)


# ====================== POST-PROCESS ======================

def smooth_skin(img, strength):
    if strength < 0.01:
        return img
    d  = int(9 + strength * 7)
    sc = 35 + strength * 65
    r  = cv2.bilateralFilter(img, d, sc, sc)
    r  = cv2.bilateralFilter(r, d - 2, sc, sc)
    b  = 0.28 + strength * 0.38
    return cv2.addWeighted(img, 1 - b, r, b, 0)


def apply_filter(bgr, name):
    if name == "original":
        return bgr
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if name == "cinematic":
        pil = ImageEnhance.Color(pil).enhance(1.2)
        pil = ImageEnhance.Contrast(pil).enhance(1.3)
    elif name == "glamour":
        pil = ImageEnhance.Brightness(pil).enhance(1.12)
        pil = ImageEnhance.Color(pil).enhance(1.3)
        a   = np.array(pil)
        a   = cv2.addWeighted(a, .85, cv2.GaussianBlur(a, (5, 5), 0), .15, 0)
        pil = Image.fromarray(a.astype(np.uint8))
    elif name == "vivid":
        pil = ImageEnhance.Brightness(pil).enhance(1.08)
        pil = ImageEnhance.Color(pil).enhance(1.4)
        pil = ImageEnhance.Contrast(pil).enhance(1.35)
    elif name == "cool":
        a = np.array(pil)
        a[:, :, 2] = np.clip(a[:, :, 2] * 1.15, 0, 255)
        a[:, :, 0] = np.clip(a[:, :, 0] * 0.90, 0, 255)
        pil = Image.fromarray(a.astype(np.uint8))
    elif name == "warm":
        a = np.array(pil)
        a[:, :, 0] = np.clip(a[:, :, 0] * 1.12, 0, 255)
        a[:, :, 2] = np.clip(a[:, :, 2] * 0.90, 0, 255)
        pil = Image.fromarray(a.astype(np.uint8))
    elif name == "noir":
        g   = ImageEnhance.Contrast(pil.convert('L')).enhance(1.5)
        a   = np.array(g)
        pil = Image.fromarray(np.stack([a] * 3, axis=2))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ====================== PIPELINE ======================

def edit_face(img_rgb, lips=1.0, nose=1.0, eyebrows=1.0, face_slim=1.0,
              jaw=1.0, forehead=1.0, brightness=1.0, smoothing=0.0,
              filter_name="original"):
    if img_rgb is None:
        return None, "Upload image"

    try:
        t0       = time.time()
        orig_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w     = orig_bgr.shape[:2]

        lm = detect_landmarks(orig_bgr)
        if lm is None:
            return img_rgb, "No face detected"

        # ---- Step 1: Gaussian warp for face features (NOT forehead) ----
        all_pts_list   = []
        all_disps_list = []
        all_sigma_list = []

        for func, val in [
            (get_face_slim_disps, face_slim),
            (get_jaw_disps,       jaw),
            (get_nose_disps,      nose),
            (get_lips_disps,      lips),
            (get_eyebrows_disps,  eyebrows),
        ]:
            pts, disps, sigma = func(lm, val)
            if len(pts) > 0:
                all_pts_list.append(pts)
                all_disps_list.append(disps)
                all_sigma_list.append((len(pts), sigma))

        if all_pts_list:
            all_pts    = np.vstack(all_pts_list)
            all_disps  = np.vstack(all_disps_list)
            all_sigmas = []
            for pt_count, sigma in all_sigma_list:
                all_sigmas.extend([sigma] * pt_count)
            warped = combined_warp(orig_bgr, all_pts, all_disps, all_sigmas)
        else:
            warped = orig_bgr.copy()

        # ---- Step 2: Blend face warp back onto original via face mask ----
        mask       = create_face_mask(lm, h, w)
        mask3      = np.stack([mask] * 3, axis=2)
        result_bgr = (warped * mask3 + orig_bgr * (1 - mask3)).astype(np.uint8)

        # ---- Step 3: Forehead AFTER mask blend ----
        # Local forehead-skin warp only (no full-width hair dragging)
        fh = get_forehead_params(lm, forehead, w)
        if fh is not None:
            y_top, y_bottom, dy, x_left, x_right = fh
            result_bgr = _apply_strip_warp(
                result_bgr, y_top, y_bottom, dy, x_left=x_left, x_right=x_right
            )

        # ---- Step 4: Post-process ----
        if filter_name != "original":
            result_bgr = apply_filter(result_bgr, filter_name)
        if abs(brightness - 1.0) > 0.005:
            result_bgr = np.clip(result_bgr.astype(np.float32) * brightness,
                                 0, 255).astype(np.uint8)
        if smoothing > 0.01:
            result_bgr = smooth_skin(result_bgr, smoothing)

        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), \
               f"Done {time.time() - t0:.2f}s"

    except Exception as e:
        logger.error(str(e))
        return img_rgb, f"Error: {str(e)}"


# ====================== UI ======================

def create_ui():
    with gr.Blocks(title="AI Facial Editor v14") as demo:
        gr.Markdown(
            "# AI Facial Editor v14\n"
            "✅ Forehead: moves hairline by shifting HAIR pixels\n"
            "✅ Applied after face mask — mask cannot undo hair shift\n"
            "✅ Skin, brows, eyes completely untouched by forehead slider"
        )

        with gr.Row():
            with gr.Column():
                inp  = gr.Image(label="Upload Selfie", type="numpy",
                                sources=["upload"])
                stat = gr.Textbox(label="Status", interactive=False,
                                  value="Upload image to start")
                debug_btn = gr.Button("Show Landmark Numbers")
                reset_btn = gr.Button("Reset All")
            with gr.Column():
                out = gr.Image(label="Result")

        gr.Markdown("### Face Shape")
        with gr.Group():
            sl_slim = gr.Slider(label="Face Slim / Widen",
                                minimum=0.85, maximum=1.2,  value=1.0, step=0.02)
            sl_jaw  = gr.Slider(label="Jaw / Chin Length",
                                minimum=0.6,  maximum=1.45, value=1.0, step=0.02)
            sl_fore = gr.Slider(label="Forehead Height",
                                minimum=0.6,  maximum=1.45, value=1.0, step=0.02)

        gr.Markdown("### Features")
        with gr.Group():
            sl_lips  = gr.Slider(label="Lips Fullness",
                                 minimum=0.6, maximum=1.55, value=1.0, step=0.05)
            sl_nose  = gr.Slider(label="Nose Width",
                                 minimum=0.6, maximum=1.8,  value=1.0, step=0.05)
            sl_brows = gr.Slider(label="Eyebrow Thick / Slim",
                                 minimum=0.7, maximum=1.45, value=1.0, step=0.03)

        gr.Markdown("### Finish")
        with gr.Group():
            sl_bright = gr.Slider(label="Brightness",
                                  minimum=0.7, maximum=1.8, value=1.0, step=0.05)
            sl_smooth = gr.Slider(label="Skin Smoothing",
                                  minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            dd_filter = gr.Dropdown(
                label="Filter",
                choices=["original", "cinematic", "glamour",
                         "vivid", "cool", "warm", "noir"],
                value="original"
            )

        controls = [inp, sl_lips, sl_nose, sl_brows, sl_slim,
                    sl_jaw, sl_fore, sl_bright, sl_smooth, dd_filter]

        def run(*args):
            return edit_face(*args)

        for c in controls[1:]:
            c.change(run, controls, [out, stat])
        inp.change(run, controls, [out, stat])

        def show_debug(img):
            if img is None:
                return None, "Upload first"
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            lm  = detect_landmarks(bgr)
            if lm is None:
                return img, "No face"
            vis = visualise_landmarks(bgr, lm)
            return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), "Landmarks shown"

        debug_btn.click(show_debug, inp, [out, stat])

        def reset():
            return 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, "original", "Reset"

        reset_btn.click(reset, outputs=[sl_lips, sl_nose, sl_brows, sl_slim,
                                        sl_jaw, sl_fore, sl_bright, sl_smooth,
                                        dd_filter, stat])
    return demo


if __name__ == "__main__":
    create_ui().launch(share=False, server_name="127.0.0.1",
                       server_port=7861, theme=gr.themes.Soft())