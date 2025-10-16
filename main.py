#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan-to-PDF (No OCR) — Text-alignment rectification + post-trim (multi-folder)
Unicode-safe paths (Korean/Hangul filenames supported)

- Processes each subfolder under the input root into one PDF named <folder>.pdf
- Paper detection: white-ish mask + scoring (used for cropping only; no perspective warp)
- Rectification: text-based skew (rotation) + horizontal shear (fix vertical lines) + vertical shear (fix horizontal lines)
- Post-trim (default ON): re-detect paper after rectification and crop margins; optional zoom-to-fill
- Unicode-safe I/O for images and PDFs (works with Korean paths on Windows)
- Safe PDF sizing (pixel → points using DPI)
- Optional downscale, sorting modes, debug overlays, per-image CSV
"""

import os
import re
import io
import cv2
import csv
import glob
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image, ExifTags
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# =============================================================================
#                               HYPERPARAMETERS
# =============================================================================
# [Paper mask – HSV]: first-stage white-ish paper candidate regions
HSV_S_MAX = 60       # upper bound for saturation (lower = more likely paper)
HSV_V_MIN = 160      # lower bound for value/brightness (higher = brighter)

# [Paper mask – LAB]: second-stage neutral/bright paper candidates
LAB_L_MIN = 170      # lower bound for L (lower this to 150–165 for dim lighting)
AB_DEV_MAX = 20      # allowed deviation of a,b from 128 (raise for warmer/cooler lighting)

# [Paper mask – Morphology]: noise removal / fill-in
MORPH_KERNEL = (7, 7)
MORPH_CLOSE_ITERS = 2
MORPH_OPEN_ITERS  = 1

# [Contour candidate filtering / scoring]
AREA_MIN_RATIO = 0.20   # contour area / image area lower bound
AREA_MAX_RATIO = 0.95   # contour area / image area upper bound
ASPECT_MIN     = 0.60   # min aspect ratio (w/h) to keep (loose for perspective)
ASPECT_MAX     = 1.90   # max aspect ratio (w/h) to keep
BORDER_MARGIN_PX   = 10     # margin near image border considered "touching"
BORDER_PENALTY_UNIT = 0.25  # penalty per touched border side (0, 0.25, 0.5)

# [Text binarization]: for line detection and projections
BIN_BLUR_KSIZE = 3            # small blur (0/1 disables)
ADAPT_BLOCK    = 31           # odd window size for adaptive threshold
ADAPT_C        = 10           # adaptive threshold constant
CLOSE_KERNEL   = (15, 1)      # connect text horizontally to strengthen text-lines

# [Skew search]: rotate to maximize row projection variance
SKEW_SEARCH_DEG   = 10.0      # search range ±deg
SKEW_SEARCH_STEP  = 0.25      # step in deg
SKEW_MIN_ABS_DEG  = 0.2       # ignore very tiny angles

# [Vertical-line shear (X)]: fix near-vertical lines (x' = x + k·y)
HOUGHP_THRESH     = 120       # HoughLinesP accumulator threshold
HOUGHP_MINLEN     = 80        # min line length (px)
HOUGHP_MAXGAP     = 10        # max gap between segments (px)
VERT_ANGLE_ALLOW  = 20.0      # treat as vertical if angle ∈ [90°±allow]
SHEAR_MIN_ABS_DEG = 0.2       # ignore very tiny shears

# [Horizontal-line shear (Y)]: fix near-horizontal lines (y' = y + k·x)
HORIZ_ANGLE_ALLOW = 20.0      # treat as horizontal if angle ∈ [0°±allow]

# [PDF sizing]: pixel→point using DPI; JPEG quality for embedding
DEFAULT_DPI   = 200
JPEG_QUALITY  = 90

# [Downscale]: limit longest side (px). 0 disables
MAX_SIDE = 3000

# [Post-trim]: trim margins after rectification (enabled by default)
POST_TRIM_PAD = 6                # keep this many pixels as safety padding
POST_TRIM_MIN_AREA_RATIO = 0.15  # ignore tiny contours when trimming
POST_TRIM_ZOOM_OVER = 4          # overshoot (px) for zoom-to-fill


# =============================================================================
#                         UNICODE-SAFE IMAGE I/O HELPERS
# =============================================================================
def imread_u(path: str) -> np.ndarray:
    """
    Unicode-safe image read for OpenCV.
    - Works even if 'path' contains Korean/Unicode characters on Windows.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def imwrite_u(path: str, img: np.ndarray) -> bool:
    """
    Unicode-safe image write for OpenCV.
    - Uses cv2.imencode + tofile() so Unicode paths work on Windows.
    """
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


# =============================================================================
#                                 SORT HELPERS
# =============================================================================
def natural_sort_key(s: str):
    """Sort '1.jpg' < '2.jpg' < '10.jpg' (natural number order)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def exif_datetime_str(path: str) -> str:
    """Return EXIF DateTimeOriginal if present, else empty string."""
    try:
        with Image.open(path) as im:
            exif = im._getexif() or {}
        if not exif:
            return ""
        tag = next((k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal"), None)
        return exif.get(tag, "") if tag else ""
    except Exception:
        return ""


# =============================================================================
#                              GEOMETRY / UTILS
# =============================================================================
def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate around center. Use black border to make later trimming easier.
    """
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

def shear_x(image: np.ndarray, shear_deg: float) -> np.ndarray:
    """
    Horizontal shear to straighten *vertical* lines.
    NOTE: measured deviation is relative to vertical; the proper coefficient is k = -tan(deg).
    """
    shear_deg = float(np.clip(shear_deg, -15.0, 15.0))
    k = -math.tan(math.radians(shear_deg))
    h, w = image.shape[:2]
    M = np.array([[1, k, 0],[0, 1, 0]], dtype=np.float32)
    new_w = int(w + abs(k)*h) + 2
    return cv2.warpAffine(
        image, M, (new_w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

def shear_y(image: np.ndarray, shear_deg: float) -> np.ndarray:
    """
    Vertical shear to straighten *horizontal* lines.
    NOTE: measured deviation is relative to horizontal; the proper coefficient is k = -tan(deg).
    """
    shear_deg = float(np.clip(shear_deg, -15.0, 15.0))
    k = -math.tan(math.radians(shear_deg))
    h, w = image.shape[:2]
    M = np.array([[1, 0, 0],[k, 1, 0]], dtype=np.float32)
    new_h = int(h + abs(k)*w) + 2
    return cv2.warpAffine(
        image, M, (w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


# =============================================================================
#                     PAPER MASK + CANDIDATE SCORING
# =============================================================================
def paper_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Build a 'paper-likelihood' mask via HSV+LAB and morphology."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, (0, 0, HSV_V_MIN), (179, HSV_S_MAX, 255))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    mask_L  = cv2.threshold(L, LAB_L_MIN, 255, cv2.THRESH_BINARY)[1]
    ab_dev  = cv2.absdiff(a, 128) + cv2.absdiff(b, 128)
    mask_ab = cv2.threshold(ab_dev, AB_DEV_MAX, 255, cv2.THRESH_BINARY_INV)[1]

    mask = cv2.bitwise_and(mask_hsv, mask_L)
    mask = cv2.bitwise_and(mask, mask_ab)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=MORPH_OPEN_ITERS)
    return mask

def _score_quad(img_bgr, contour, quad_pts, mask):
    """
    Score candidate quadrilaterals by area ratio, brightness, 'paper mask' ratio,
    rectangularity, and border-touch penalty.
    """
    H, W = img_bgr.shape[:2]
    area_img = float(W*H)
    area_cnt = cv2.contourArea(contour)
    area_ratio = area_cnt / max(area_img,1.0)

    rect_mask = np.zeros((H,W), np.uint8)
    cv2.fillConvexPoly(rect_mask, quad_pts.astype(np.int32), 255)
    mean_v = cv2.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:,:,2], mask=rect_mask)[0]/255.0

    x,y,w,h = cv2.boundingRect(quad_pts.astype(np.int32))
    rectangularity = area_cnt / max(w*h,1.0)

    margin = BORDER_MARGIN_PX
    xs, ys = quad_pts[:,0], quad_pts[:,1]
    touches = 0
    if (xs < margin).any() or (xs > W-margin).any(): touches += 1
    if (ys < margin).any() or (ys > H-margin).any(): touches += 1
    border_penalty = BORDER_PENALTY_UNIT * touches

    mask_inside = cv2.mean(mask, mask=rect_mask)[0]/255.0

    score = area_ratio * (0.5*mean_v + 0.5*mask_inside) * rectangularity * (1.0 - border_penalty)
    detail = {"area_ratio":area_ratio,"mean_v":mean_v,"mask_inside":mask_inside,
              "rectangularity":rectangularity,"border_penalty":border_penalty}
    return float(score), detail

def detect_paper_quad(img_bgr: np.ndarray):
    """
    Return (quad_pts in order tl,tr,br,bl, detail) or (None, reason).
    Used only for cropping — no perspective warp.
    """
    H, W = img_bgr.shape[:2]
    mask = paper_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"reason":"no contours from mask"}

    best = (None, -1e9, None)
    for c in contours:
        area = cv2.contourArea(c)
        r = area / float(W*H)
        if r < AREA_MIN_RATIO or r > AREA_MAX_RATIO:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4:
            rect = cv2.minAreaRect(c)
            approx = cv2.boxPoints(rect).reshape(-1,1,2).astype(np.int32)

        quad = approx.reshape(4,2).astype(np.float32)
        if not cv2.isContourConvex(approx):
            continue

        s = quad.sum(axis=1); diff = np.diff(quad,axis=1).ravel()
        tl = quad[np.argmin(s)]; br = quad[np.argmax(s)]
        tr = quad[np.argmin(diff)]; bl = quad[np.argmax(diff)]
        quad = np.array([tl,tr,br,bl], np.float32)

        wA = np.linalg.norm(br-bl); wB = np.linalg.norm(tr-tl)
        hA = np.linalg.norm(tr-br); hB = np.linalg.norm(tl-bl)
        maxW, maxH = max(wA,wB), max(hA,hB)
        ratio = maxW / max(maxH,1.0)
        if not (ASPECT_MIN <= ratio <= ASPECT_MAX):
            continue

        score, detail = _score_quad(img_bgr, c, quad, mask)
        if score > best[1]:
            best = (quad, score, detail)

    if best[0] is None:
        return None, {"reason":"no good quad", "n":len(contours)}
    return best[0], {"score":best[1], **best[2]}


# =============================================================================
#                         TEXT BINARY & RECTIFICATION
# =============================================================================
def text_binary(img_bgr: np.ndarray) -> np.ndarray:
    """Binary image emphasizing text lines for projections/Hough."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if BIN_BLUR_KSIZE and BIN_BLUR_KSIZE > 1:
        k = BIN_BLUR_KSIZE + (BIN_BLUR_KSIZE % 2 == 0)
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPT_BLOCK | 1, ADAPT_C
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bin_img

def estimate_skew_angle(bin_img: np.ndarray) -> float:
    """
    Search for rotation angle (deg) maximizing row-projection variance.
    """
    h, w = bin_img.shape
    best_angle = 0.0
    best_score = -1e18
    angles = np.arange(-SKEW_SEARCH_DEG, SKEW_SEARCH_DEG + 1e-6, SKEW_SEARCH_STEP)
    for ang in angles:
        M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
        rot = cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        proj = np.sum(rot > 0, axis=1).astype(np.float32)
        score = np.var(proj)
        if score > best_score:
            best_score = score
            best_angle = ang
    return float(best_angle)

def estimate_vertical_shear_deg(bin_img_rot: np.ndarray) -> float:
    """
    Return deviation from vertical (deg) using near-vertical lines from Hough.
    Positive means lines lean toward +x as y increases.
    """
    lines = cv2.HoughLinesP(
        bin_img_rot, 1, np.pi/180, threshold=HOUGHP_THRESH,
        minLineLength=HOUGHP_MINLEN, maxLineGap=HOUGHP_MAXGAP
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:,0]:
        dx = x2 - x1; dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = math.degrees(math.atan2(dy, dx))  # 0=horiz, 90=vertical
        if 90 - VERT_ANGLE_ALLOW <= abs(ang) <= 90 + VERT_ANGLE_ALLOW:
            dev = (90 - abs(ang)) * (1 if ang >= 0 else -1)
            angles.append(dev)
    if not angles:
        return 0.0
    return float(np.median(angles))

def estimate_horizontal_shear_deg(bin_img_rot: np.ndarray) -> float:
    """
    Return deviation from horizontal (deg) using near-horizontal lines from Hough.
    Positive means lines tilt upward with +x.
    """
    lines = cv2.HoughLinesP(
        bin_img_rot, 1, np.pi/180,
        threshold=HOUGHP_THRESH, minLineLength=HOUGHP_MINLEN, maxLineGap=HOUGHP_MAXGAP
    )
    if lines is None:
        return 0.0

    devs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1; dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = math.degrees(math.atan2(dy, dx))  # 0 = horizontal
        if ang > 90:  ang -= 180
        if ang < -90: ang += 180
        if abs(ang) <= HORIZ_ANGLE_ALLOW:
            devs.append(ang)
    if not devs:
        return 0.0
    return float(np.median(devs))


# =============================================================================
#                    CROP (NO WARP) + TEXT RECTIFICATION
# =============================================================================
def crop_and_text_rectify(
    image_path: str,
    save_debug: bool = False,
    debug_dir: str = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pipeline for a single image:
      1) Use paper quad for tight crop only (no perspective warp)
      2) Text-based rectification:
         a) skew (rotation),
         b) horizontal shear to fix vertical lines,
         c) vertical shear to fix horizontal lines
      3) Force portrait orientation at the end
    """
    img = imread_u(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    H0, W0 = img.shape[:2]

    # Paper quad → tight crop only
    quad, detail = detect_paper_quad(img)
    if quad is not None:
        xs = quad[:,0]; ys = quad[:,1]
        x0, y0 = max(int(xs.min()), 0), max(int(ys.min()), 0)
        x1, y1 = min(int(xs.max()), W0-1), min(int(ys.max()), H0-1)
        cropped = img[y0:y1+1, x0:x1+1]
        crop_box = (x0,y0,x1,y1)
    else:
        cropped = img.copy()
        crop_box = None

    # 1) Skew (rotation)
    bin0 = text_binary(cropped)
    ang = estimate_skew_angle(bin0)
    rotated = False
    img1 = cropped
    if abs(ang) >= SKEW_MIN_ABS_DEG:
        img1 = rotate_image(cropped, -ang)
        rotated = True

    # 2) Horizontal shear (fix vertical lines)
    bin1 = text_binary(img1)
    shear_x_deg = estimate_vertical_shear_deg(bin1)
    sheared_x = False
    img2 = img1
    if abs(shear_x_deg) >= SHEAR_MIN_ABS_DEG:
        img2 = shear_x(img1, shear_x_deg)
        sheared_x = True

    # 3) Vertical shear (fix horizontal lines)
    bin2 = text_binary(img2)
    shear_y_deg = estimate_horizontal_shear_deg(bin2)
    sheared_y = False
    img3 = img2
    if abs(shear_y_deg) >= SHEAR_MIN_ABS_DEG:
        img3 = shear_y(img2, shear_y_deg)
        sheared_y = True

    # Portrait orientation
    final_img = img3
    if final_img.shape[1] > final_img.shape[0]:
        final_img = rotate_image(final_img, 90)

    # Debug artifacts (unicode-safe writes)
    if save_debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        overlay = img.copy()
        if quad is not None:
            poly = quad.astype(np.int32).reshape(-1,1,2)
            cv2.polylines(overlay, [poly], True, (0,255,0), 3)
        if crop_box:
            x0,y0,x1,y1 = crop_box
            cv2.rectangle(overlay, (x0,y0),(x1,y1),(255,0,0),2)
        imwrite_u(os.path.join(debug_dir, f"{os.path.basename(image_path)}_overlay.jpg"), overlay)
        imwrite_u(os.path.join(debug_dir, f"{os.path.basename(image_path)}_crop.jpg"), cropped)
        imwrite_u(os.path.join(debug_dir, f"{os.path.basename(image_path)}_rot.jpg"), img1)
        imwrite_u(os.path.join(debug_dir, f"{os.path.basename(image_path)}_shearx.jpg"), img2)
        imwrite_u(os.path.join(debug_dir, f"{os.path.basename(image_path)}_sheary.jpg"), img3)

    debug_info = {
        "filename": os.path.basename(image_path),
        "orig_w": W0, "orig_h": H0,
        "crop_box": crop_box,
        "quad_found": quad is not None,
        **(detail if quad is not None else {}),
        "skew_deg": round(float(ang),2), "rotated": rotated,
        "shear_x_deg": round(float(shear_x_deg),2), "sheared_x": sheared_x,
        "shear_y_deg": round(float(shear_y_deg),2), "sheared_y": sheared_y,
        "out_w": final_img.shape[1], "out_h": final_img.shape[0]
    }
    return final_img, debug_info


# =============================================================================
#                              OPTIONAL POST-TRIM
# =============================================================================
def post_trim_margins(
    img_bgr: np.ndarray,
    pad: int = POST_TRIM_PAD,
    zoom_to_fill: bool = True,
    zoom_overshoot_px: int = POST_TRIM_ZOOM_OVER
) -> Tuple[np.ndarray, Tuple[int,int,int,int] or None]:
    """
    Trim margins after rectification using paper mask again, then optionally
    'zoom-to-fill' so the paper slightly overfills the frame.

    Steps:
      1) Detect max external contour from paper mask
      2) Tight bounding-rect crop with 'pad' pixels kept
      3) If zoom_to_fill: scale uniformly so the contour bbox exceeds the frame
         by 'zoom_overshoot_px', then center-crop back to original frame size
    """
    H, W = img_bgr.shape[:2]

    # Try paper mask first
    mask = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # If the image already has black borders from previous warps, paper_mask() is better:
    mask = paper_mask(img_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fallback: use non-black bounding box (accounts for black borders from warp)
    if not contours:
        nz = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) > 5
        ys, xs = np.where(nz)
        if len(xs) == 0:
            return img_bgr, None
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
        x0 = max(x - pad, 0); y0 = max(y - pad, 0)
        x1 = min(x + w + pad, W); y1 = min(y + h + pad, H)
        cropped = img_bgr[y0:y1, x0:x1]
        return cropped, (x0,y0,x1,y1)

    # Largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    c_main = contours[0]
    x, y, w, h = cv2.boundingRect(c_main)
    if w*h < POST_TRIM_MIN_AREA_RATIO * (W*H):
        return img_bgr, None

    # Tight crop with padding
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + w + pad, W); y1 = min(y + h + pad, H)
    cropped = img_bgr[y0:y1, x0:x1]

    if not zoom_to_fill:
        return cropped, (x0,y0,x1,y1)

    # Zoom-to-fill: scale so bbox exceeds frame slightly, then center-crop
    c_local = c_main.reshape(-1, 2).astype(np.int32)
    c_local[:, 0] -= x0
    c_local[:, 1] -= y0
    Hc, Wc = cropped.shape[:2]

    cx, cy, cw, ch = cv2.boundingRect(c_local)
    target_w = Wc + 2*zoom_overshoot_px
    target_h = Hc + 2*zoom_overshoot_px
    sw = target_w / max(cw, 1)
    sh = target_h / max(ch, 1)
    s = max(sw, sh)

    if s <= 1.0:
        return cropped, (x0,y0,x1,y1)

    new_w = int(round(Wc * s))
    new_h = int(round(Hc * s))
    big = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    cx0 = new_w // 2
    cy0 = new_h // 2
    xs = max(0, cx0 - Wc // 2); ys = max(0, cy0 - Hc // 2)
    xe = min(new_w, xs + Wc); ye = min(new_h, ys + Hc)
    cropped_zoom = big[ys:ye, xs:xe]

    # If off by 1–2 pixels, pad with black
    if cropped_zoom.shape[0] != Hc or cropped_zoom.shape[1] != Wc:
        canvas = np.zeros((Hc, Wc, 3), dtype=big.dtype)
        chh, cww = cropped_zoom.shape[:2]
        canvas[:chh, :cww] = cropped_zoom
        cropped_zoom = canvas

    return cropped_zoom, (x0,y0,x1,y1)


# =============================================================================
#                           DOWNSCALE & WRITE PDF
# =============================================================================
def downscale_if_needed(img_bgr: np.ndarray, max_side: int = MAX_SIDE) -> Tuple[np.ndarray, float]:
    """Downscale longest side to <= max_side. Return (image, scale)."""
    if max_side <= 0:
        return img_bgr, 1.0
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr, 1.0
    scale = max_side / float(m)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), float(scale)

def images_to_single_pdf(
    images_bgr: List[np.ndarray],
    output_pdf_path: str,
    dpi: int = DEFAULT_DPI,
    jpeg_quality: int = JPEG_QUALITY
) -> List[Dict[str, Any]]:
    """
    Embed each image as a full-bleed JPEG page with pixel→point sizing.
    Unicode-safe path: open a binary handle and pass it to reportlab.Canvas.
    """
    if not images_bgr:
        raise ValueError("No images to write.")

    # Ensure parent directory exists
    out_dir = os.path.dirname(output_pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    info = []
    # Open as a binary file handle so non-ASCII paths work across platforms
    with open(output_pdf_path, "wb") as fh:
        c = canvas.Canvas(fh)

        for img in images_bgr:
            h, w = img.shape[:2]
            pw = max(int(round(w * 72.0 / dpi)), 1)
            ph = max(int(round(h * 72.0 / dpi)), 1)
            c.setPageSize((pw, ph))

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=int(np.clip(jpeg_quality,1,100)))
            buf.seek(0)
            c.drawImage(ImageReader(buf), 0, 0, width=pw, height=ph)
            c.showPage()

            info.append({"page_w_pt":pw,"page_h_pt":ph})
        c.save()
    return info


# =============================================================================
#                                 PIPELINE
# =============================================================================
def collect_images(input_folder: str, patterns: Tuple[str, ...], sort_mode: str) -> List[str]:
    """Collect image paths by glob patterns and sort by the selected mode."""
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(input_folder, pat)))
    if sort_mode == "natural":
        paths.sort(key=natural_sort_key)
    elif sort_mode == "lex":
        paths.sort()
    elif sort_mode == "mtime":
        paths.sort(key=os.path.getmtime)
    elif sort_mode == "exif":
        paths.sort(key=exif_datetime_str)
    else:
        paths.sort()
    return paths

def process_folder_to_pdf(
    input_folder: str,
    output_pdf_path: str,
    patterns: Tuple[str, ...] = ("*.jpg","*.jpeg","*.png"),
    sort_mode: str = "natural",
    dpi: int = DEFAULT_DPI,
    max_side: int = MAX_SIDE,
    jpeg_quality: int = JPEG_QUALITY,
    debug: bool = False,
    debug_dir: str = None,
    csv_path: str = None,
    post_trim: bool = True,
    post_trim_pad: int = POST_TRIM_PAD,
    post_trim_zoom: bool = True,
    post_trim_zoom_over: int = POST_TRIM_ZOOM_OVER
) -> str:
    """
    Process a single *image folder* into one PDF.
    """
    img_paths = collect_images(input_folder, patterns, sort_mode)
    if not img_paths:
        raise FileNotFoundError(f"No images in {input_folder} with {patterns}")

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    pages: List[np.ndarray] = []
    rows: List[Dict[str,Any]] = []

    for i, p in enumerate(img_paths, 1):
        rectified, d = crop_and_text_rectify(p, save_debug=debug, debug_dir=debug_dir)

        # Optional post-trim & zoom-to-fill
        trim_box = None
        if post_trim:
            rectified, trim_box = post_trim_margins(
                rectified,
                pad=post_trim_pad,
                zoom_to_fill=post_trim_zoom,
                zoom_overshoot_px=post_trim_zoom_over
            )
            if debug and debug_dir and trim_box is not None:
                vis = rectified.copy()
                cv2.rectangle(vis, (0, 0), (vis.shape[1]-1, vis.shape[0]-1), (0, 0, 255), 2)
                imwrite_u(os.path.join(debug_dir, f"{os.path.basename(p)}_posttrim.jpg"), vis)

        # Downscale after trimming (if any)
        rectified, scale = downscale_if_needed(rectified, max_side=max_side)
        pages.append(rectified)

        # Collect per-image info (CSV optional)
        rows.append({
            "filename": d["filename"],
            "orig_w": d["orig_w"], "orig_h": d["orig_h"],
            "quad_found": d["quad_found"], "crop_box": d["crop_box"],
            "skew_deg": d["skew_deg"], "shear_x_deg": d["shear_x_deg"], "shear_y_deg": d["shear_y_deg"],
            "out_w": rectified.shape[1], "out_h": rectified.shape[0],
            "downscale": round(scale,4),
            "post_trim_box": trim_box,
            **({k:d[k] for k in ("score","area_ratio","mask_inside","rectangularity","border_penalty") if k in d})
        })

        print(f"[{i}/{len(img_paths)}] {os.path.basename(p)} | skew={d['skew_deg']}° "
              f"sx={d['shear_x_deg']}° sy={d['shear_y_deg']}° "
              f"posttrim={'Y' if trim_box else 'N'} "
              f"scale={round(scale,4)} "
              f"out={rectified.shape[1]}x{rectified.shape[0]}")

    # Write PDF (unicode-safe path)
    images_to_single_pdf(pages, output_pdf_path, dpi=dpi, jpeg_quality=jpeg_quality)

    # Optional CSV (per-folder)
    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fields = ["filename","orig_w","orig_h","quad_found","crop_box",
                  "skew_deg","shear_x_deg","shear_y_deg","out_w","out_h",
                  "downscale","post_trim_box","score","area_ratio","mask_inside","rectangularity","border_penalty"]
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig for Excel-friendly BOM on Windows
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"✅ Done: {output_pdf_path}")
    return output_pdf_path


def process_root_multiple_folders(
    input_root: str,
    output_dir: str,
    patterns: Tuple[str, ...],
    csv_template: str = None,
    **kwargs
):
    """
    Iterate subfolders of the input root and produce one PDF per subfolder.
    The PDF filename is <subfolder>.pdf in the output directory.
    CSV (if requested) can be created per folder with a suffix.
    """
    os.makedirs(output_dir, exist_ok=True)
    subdirs = [d for d in sorted(os.listdir(input_root)) if os.path.isdir(os.path.join(input_root, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found in {input_root}")

    def csv_path_for(folder_name: str):
        if not csv_template:
            return None
        base, ext = os.path.splitext(csv_template)
        return f"{base}_{folder_name}{ext or '.csv'}"

    created = []
    for d in subdirs:
        in_dir = os.path.join(input_root, d)
        out_pdf = os.path.join(output_dir, f"{d}.pdf")
        print(f"\n=== Processing folder: {d} ===")
        try:
            process_folder_to_pdf(
                in_dir,
                out_pdf,
                patterns=patterns,
                csv_path=csv_path_for(d),
                **kwargs
            )
            created.append(out_pdf)
        except FileNotFoundError as e:
            print(f"  (skip) {d}: {e}")
    print(f"\nAll done. Created {len(created)} PDFs under: {output_dir}")
    return created


# =============================================================================
#                                      CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Scan-to-PDF (No OCR) — text-alignment + post-trim (multi-folder, Unicode-safe paths)")
    ap.add_argument("-i","--input", required=True, help="Input ROOT folder (contains subfolders)")
    ap.add_argument("-O","--output-dir", required=True, help="Output directory to write PDFs into")
    ap.add_argument("--ext", default="jpg,jpeg,png", help="Comma-separated extensions")
    ap.add_argument("--sort", default="natural", choices=["natural","lex","mtime","exif"], help="Sorting mode")

    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="DPI used for pixel→point conversion")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE, help="Downscale longest side (0=off)")
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY, help="JPEG quality 1–100")

    ap.add_argument("--debug", action="store_true", help="Save debug overlays/images")
    ap.add_argument("--debug-dir", default=None, help="Folder to save debug images")
    ap.add_argument("--csv", default=None, help="CSV path template; per-folder suffix is appended automatically")

    # Defaults: post-trim and zoom enabled; use --no-... to disable
    ap.add_argument("--no-post-trim", dest="post_trim", action="store_false",
                    help="Disable post-trim")
    ap.add_argument("--no-post-trim-zoom", dest="post_trim_zoom", action="store_false",
                    help="Disable zoom-to-fill after post-trim")
    ap.add_argument("--post-trim-pad", type=int, default=POST_TRIM_PAD, help="Padding(px) to keep after post-trim")
    ap.add_argument("--post-trim-zoom-over", type=int, default=POST_TRIM_ZOOM_OVER,
                    help="Overshoot(px) for zoom-to-fill (0–8 recommended)")

    # Set defaults to True for these booleans
    ap.set_defaults(post_trim=True, post_trim_zoom=True)
    return ap.parse_args()

def main():
    args = parse_args()
    patterns = tuple(f"*.{e.strip().lstrip('.')}" for e in args.ext.split(",") if e.strip())

    process_root_multiple_folders(
        input_root=args.input,
        output_dir=args.output_dir,
        patterns=patterns,
        csv_template=args.csv,
        sort_mode=args.sort,
        dpi=args.dpi,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        debug=args.debug,
        debug_dir=args.debug_dir,
        post_trim=args.post_trim,
        post_trim_pad=args.post_trim_pad,
        post_trim_zoom=args.post_trim_zoom,
        post_trim_zoom_over=args.post_trim_zoom_over
    )

if __name__ == "__main__":
    main()
