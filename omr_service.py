#!/usr/bin/env python3
"""
QuickGrader OMR Service v4.0 (TNMaker-style stability)
- Robust corner marker detection (square marker template + edge match in 4 corners)
- Perspective warp to fixed canvas
- Read Student ID (3 digits) + Answers (10 questions, A/B/C/D)
- Optional debug overlay image (base64 jpeg) to verify alignment
"""

import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==================== CONFIG ====================
WARP_W, WARP_H = 900, 1300

# Marker match
CORNER_SEARCH_FRAC = 0.38  # search area in each corner
TPL_SIZE = 90              # synthetic marker template size
SCALES = np.linspace(0.75, 1.45, 15)  # template scales to try
EDGE1, EDGE2 = 50, 150

# Quality
BLUR_THRESHOLD = 25.0

# Bubble detection thresholds (relative)
# (đừng đặt quá cao vì HS tô nhạt sẽ rớt)
MIN_FILLED_SCORE = 0.080
MIN_GAP_SCORE = 0.020

CHOICES = ["A", "B", "C", "D"]

# ==================== ROI for YOUR SHEET ====================
# These ROI values are normalized on the warped canvas (WARP_W x WARP_H)
# Based on your provided sheet photo (ID block top-left, Answer block bottom-left)

# Tight ROI just around the 3x10 ID bubbles (avoid headers/text)
ID_BUBBLE_ROI = (0.105, 0.275, 0.405, 0.635)     # (x1,y1,x2,y2)

# Tight ROI around the A/B/C/D bubbles (exclude "Câu ..." text as much as possible)
ANS_BUBBLE_ROI = (0.155, 0.585, 0.875, 0.930)    # (x1,y1,x2,y2)

TOTAL_QUESTIONS_DEFAULT = 10

# ==================== UTILS ====================
def b64_to_img(b64: str):
    if not b64:
        return None
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def check_blur(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def norm_roi(img, roi):
    x1n, y1n, x2n, y2n = roi
    h, w = img.shape[:2]
    x1, y1 = int(x1n * w), int(y1n * h)
    x2, y2 = int(x2n * w), int(y2n * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

def jpeg_b64(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# ==================== MARKER TEMPLATE MATCH ====================
def gen_marker_template(size=TPL_SIZE, hole_ratio=0.35):
    """
    Synthetic marker like your sheet: black square with white inner square.
    We'll match on edges so lighting doesn't matter much.
    """
    tpl = np.ones((size, size), dtype=np.uint8) * 255
    cv2.rectangle(tpl, (0, 0), (size - 1, size - 1), 0, -1)
    hole = int(size * hole_ratio)
    s = (size - hole) // 2
    cv2.rectangle(tpl, (s, s), (s + hole, s + hole), 255, -1)
    tpl_edges = cv2.Canny(tpl, EDGE1, EDGE2)
    return tpl_edges

def find_marker_in_corner(gray, corner, tpl_edges):
    """
    Search for marker in a corner region using multi-scale edge template matching.
    Returns: ((x,y), score, (tw,th)) where (x,y) is top-left of match in FULL image.
    """
    H, W = gray.shape
    rw, rh = int(W * CORNER_SEARCH_FRAC), int(H * CORNER_SEARCH_FRAC)

    if corner == "tl":
        x0, y0 = 0, 0
    elif corner == "tr":
        x0, y0 = W - rw, 0
    elif corner == "bl":
        x0, y0 = 0, H - rh
    else:  # br
        x0, y0 = W - rw, H - rh

    roi = gray[y0:y0 + rh, x0:x0 + rw]
    roi_edges = cv2.Canny(roi, EDGE1, EDGE2)

    best_loc = None
    best_score = -1.0
    best_size = None

    for s in SCALES:
        th = int(tpl_edges.shape[1] * s)
        tv = int(tpl_edges.shape[0] * s)
        if th < 28 or tv < 28:
            continue
        if th >= roi_edges.shape[1] or tv >= roi_edges.shape[0]:
            continue

        tpl_rs = cv2.resize(tpl_edges, (th, tv), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(roi_edges, tpl_rs, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv2.minMaxLoc(res)

        if maxv > best_score:
            best_score = float(maxv)
            best_loc = (x0 + maxl[0], y0 + maxl[1])
            best_size = (th, tv)

    return best_loc, best_score, best_size

def find_4_markers(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tpl_edges = gen_marker_template()

    found = {}
    for c in ["tl", "tr", "bl", "br"]:
        loc, score, sz = find_marker_in_corner(gray, c, tpl_edges)
        found[c] = {"loc": loc, "score": score, "size": sz}

    # validate
    if any(found[k]["loc"] is None for k in found):
        return None, {"markers_found": False, "found": found}

    # Build centers
    centers = {}
    for k in found:
        (x, y) = found[k]["loc"]
        (tw, th) = found[k]["size"]
        centers[k] = (x + tw / 2.0, y + th / 2.0)

    return centers, {"markers_found": True, "found": found, "centers": centers}

def warp_by_marker_centers(img_bgr, centers):
    """
    Warp using marker CENTERS to fixed positions inside the warped canvas.
    This is stable across printing/cropping.
    """
    m = 60  # margin inside warp for marker centers
    src = np.array([centers["tl"], centers["tr"], centers["br"], centers["bl"]], dtype=np.float32)
    dst = np.array([[m, m], [WARP_W - m, m], [WARP_W - m, WARP_H - m], [m, WARP_H - m]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (WARP_W, WARP_H))
    return warped

# ==================== BUBBLE SCORING ====================
def illumination_normalize(gray):
    # Flat-field normalization to reduce shadows
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    norm = cv2.divide(gray, blur, scale=255)
    return norm

def bubble_score(gray_norm, cx, cy, r):
    """
    TNMaker-like: compare inner area darkness to surrounding background.
    Score higher => more likely filled.
    """
    h, w = gray_norm.shape
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    pad = int(r * 2.2)

    x1, x2 = max(0, cx - pad), min(w, cx + pad)
    y1, y2 = max(0, cy - pad), min(h, cy + pad)
    roi = gray_norm[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    ccx, ccy = roi.shape[1] // 2, roi.shape[0] // 2
    dist = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2)

    inner = dist <= r * 0.55
    bg = (dist >= r * 1.20) & (dist <= r * 1.75)

    if inner.sum() < 20 or bg.sum() < 30:
        return 0.0

    inner_mean = float(np.mean(roi[inner]))
    bg_mean = float(np.mean(roi[bg]))
    # filled => inner darker => lower mean => higher (bg - inner)
    score = (bg_mean - inner_mean) / 255.0
    return max(0.0, float(score))

def build_grid_scores(gray_norm, rows, cols):
    h, w = gray_norm.shape
    dx, dy = w / cols, h / rows
    r = int(min(dx, dy) * 0.22)
    r = max(7, min(18, r))

    scores = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) * dx
            cy = (i + 0.5) * dy
            scores[i, j] = bubble_score(gray_norm, cx, cy, r)
    return scores, r

def pick_one(scores_row, min_score=MIN_FILLED_SCORE, min_gap=MIN_GAP_SCORE):
    best = int(np.argmax(scores_row))
    sorted_s = np.sort(scores_row)[::-1]
    top = float(sorted_s[0])
    second = float(sorted_s[1]) if len(sorted_s) > 1 else 0.0
    if top < min_score:
        return None
    if (top - second) < min_gap:
        return None
    return best

# ==================== READ ID + ANSWERS ====================
def read_student_id(warped_bgr, debug=False):
    roi_img, roi_box = norm_roi(warped_bgr, ID_BUBBLE_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    gray_n = illumination_normalize(gray)

    scores, r = build_grid_scores(gray_n, rows=10, cols=3)

    digits = []
    for col in range(3):
        sel = pick_one(scores[:, col], min_score=0.070, min_gap=0.015)
        digits.append(sel)

    dbg = {"roi_box": roi_box, "radius": r}
    if debug:
        dbg["scores"] = scores.tolist()
        dbg["digits"] = digits

    if any(d is None for d in digits):
        return None, dbg

    sid_raw = "".join(str(d) for d in digits)  # rows represent 0..9 top->bottom
    try:
        sid = str(int(sid_raw))  # remove leading zeros
    except Exception:
        sid = sid_raw
    return sid, dbg

def read_answers(warped_bgr, total_q=TOTAL_QUESTIONS_DEFAULT, debug=False):
    roi_img, roi_box = norm_roi(warped_bgr, ANS_BUBBLE_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    gray_n = illumination_normalize(gray)

    scores, r = build_grid_scores(gray_n, rows=total_q, cols=4)

    answers = []
    picks = []
    for i in range(total_q):
        sel = pick_one(scores[i, :], min_score=0.080, min_gap=0.020)
        picks.append(sel)
        answers.append(CHOICES[sel] if sel is not None else None)

    dbg = {"roi_box": roi_box, "radius": r}
    if debug:
        dbg["scores"] = scores.tolist()
        dbg["picks"] = picks
    return answers, dbg

def grade(student_ans, key, threshold):
    total = len(key)
    score = 0
    for i in range(total):
        a = (student_ans[i] or "").strip().upper()
        k = (str(key[i]) or "").strip().upper()
        if a and a == k:
            score += 1
    pct = int(round((score / total) * 100)) if total > 0 else 0
    status = "PASS" if pct >= int(threshold) else "FAIL"
    return score, pct, status

def make_debug_overlay(warped, sid_dbg, ans_dbg, sid, answers):
    """
    Draw ROIs and selected cells to visually verify.
    """
    out = warped.copy()

    # Draw ROIs
    for name, (x1,y1,x2,y2) in [
        ("ID", sid_dbg.get("roi_box", (0,0,0,0))),
        ("ANS", ans_dbg.get("roi_box", (0,0,0,0))),
    ]:
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, name, (x1+8, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Mark chosen ID bubbles (approx centers)
    try:
        (x1,y1,x2,y2) = sid_dbg["roi_box"]
        w = x2-x1; h = y2-y1
        cols, rows = 3, 10
        dx, dy = w/cols, h/rows
        r = int(sid_dbg.get("radius", 12))
        if sid:
            # rebuild digits with leading zeros by scores argmax if available
            digits = sid_dbg.get("digits", None)
            if digits and len(digits)==3:
                for j, d in enumerate(digits):
                    if d is None: continue
                    cx = int(x1 + (j+0.5)*dx)
                    cy = int(y1 + (d+0.5)*dy)
                    cv2.circle(out, (cx,cy), r, (0,0,255), 2)
    except Exception:
        pass

    # Mark chosen answers
    try:
        (x1,y1,x2,y2) = ans_dbg["roi_box"]
        w = x2-x1; h = y2-y1
        cols, rows = 4, len(answers)
        dx, dy = w/cols, h/rows
        r = int(ans_dbg.get("radius", 12))
        for i,a in enumerate(answers):
            if not a: continue
            j = CHOICES.index(a)
            cx = int(x1 + (j+0.5)*dx)
            cy = int(y1 + (i+0.5)*dy)
            cv2.circle(out, (cx,cy), r, (255,0,0), 2)
    except Exception:
        pass

    return out

# ==================== API ====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "QuickGrader OMR v4.0"}), 200

@app.route("/process_omr", methods=["POST"])
def process_omr():
    try:
        data = request.json or {}
        image_data = data.get("image") or data.get("image_base64")
        answer_key = data.get("answer_key", [])
        total_q = int(data.get("total_questions", len(answer_key) or TOTAL_QUESTIONS_DEFAULT))
        threshold = int(data.get("pass_threshold", 80))
        debug = bool(data.get("debug", False))

        if not image_data:
            return jsonify({"success": False, "error": "Missing image"}), 400
        if not answer_key:
            return jsonify({"success": False, "error": "Missing answer_key"}), 400

        img = b64_to_img(image_data)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        blur_var = check_blur(img)
        if blur_var < BLUR_THRESHOLD:
            return jsonify({
                "success": False,
                "error": "Image too blurry",
                "blur_variance": round(blur_var, 2)
            }), 422

        centers, m_dbg = find_4_markers(img)
        if centers is None:
            return jsonify({"success": False, "error": "Cannot find 4 markers", "debug": m_dbg}), 422

        warped = warp_by_marker_centers(img, centers)

        student_id, sid_dbg = read_student_id(warped, debug=debug)
        answers, ans_dbg = read_answers(warped, total_q=total_q, debug=debug)

        if student_id is None:
            # still return answers; but flag warning
            student_id = ""
            sid_warn = True
        else:
            sid_warn = False

        score, pct, status = grade(answers, answer_key, threshold)

        resp = {
            "success": True,
            "student_id": str(student_id),
            "student_name": f"Hoc sinh {student_id}" if student_id else "",
            "answers": [a if a else "" for a in answers],
            "score": score,
            "percentage": pct,
            "status": status,
            "warnings": []
        }
        if sid_warn:
            resp["warnings"].append("Không đọc được mã HS (ID)")

        if debug:
            overlay = make_debug_overlay(warped, sid_dbg, ans_dbg, student_id, answers)
            resp["debug"] = {
                "marker": m_dbg,
                "student_id": sid_dbg,
                "answers": ans_dbg,
                "overlay_jpg_base64": jpeg_b64(overlay, quality=85)
            }

        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
