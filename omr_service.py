#!/usr/bin/env python3
"""
QuickGrader OMR Service v3.1 (Fix phiếu 10 câu + ID 3 số)
- Warp theo 4 marker góc
- Đọc ID theo ROI chuẩn (3 cột x 10 hàng)
- Đọc đáp án bằng phát hiện bong bóng (HoughCircles) -> gom 10 hàng x 4 cột
"""

import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==================== CẤU HÌNH ====================
WARP_W, WARP_H = 900, 1300

BLUR_THRESHOLD = 35.0

# Marker (ô vuông đen ở 4 góc)
MARKER_MIN_AREA = 0.0002
MARKER_MAX_AREA = 0.12
MARKER_MIN_CIRC = 0.38

# Ngưỡng tô (có thể tinh chỉnh nếu giấy in nhạt/đậm)
FILL_THRESHOLD = 0.12     # tối thiểu coi là “tô”
MIN_GAP = 0.04            # chênh lệch tối thiểu giữa tô mạnh nhất và mạnh nhì

CHOICES = ["A", "B", "C", "D"]

# ==================== ROI CHUẨN THEO PHIẾU BẠN (ẢNH BẠN GỬI) ====================
# Lưu ý: ROI là theo ảnh đã warp về 900x1300 (tỉ lệ 0..1)
# 1) Vùng ID (chỉ lấy khu bong bóng 3 cột trăm/chục/đơn vị)
STUDENT_ID_ROI = (0.08, 0.22, 0.42, 0.56)   # (x1, y1, x2, y2)

# 2) Vùng đáp án 10 câu: lấy cả khu bong bóng, bỏ bớt phần header phía trên để tránh “dính” ID
# (ROI này vẫn còn chữ Câu/A/B/C/D nhưng ta đọc bằng HoughCircles nên vẫn ổn)
ANSWERS_ROI = (0.05, 0.60, 0.72, 0.92)

# Grid ID
STUDENT_COLS, STUDENT_ROWS = 3, 10

# ==================== CORE FUNCTIONS ====================
def decode_img(b64: str):
    if not b64:
        return None
    if "," in b64:
        b64 = b64.split(",")[1]
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

def check_blur(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def order_pts(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    # tl, tr, br, bl
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype="float32"
    )

def find_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    area = gray.size
    cands = []

    methods = [
        lambda: cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 7),
        lambda: cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        lambda: cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5),
    ]

    for m in methods:
        try:
            th = m()
            cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if area * MARKER_MIN_AREA < a < area * MARKER_MAX_AREA:
                    p = cv2.arcLength(cnt, True)
                    if p > 0:
                        circ = 4 * np.pi * (a / (p * p))
                        if circ >= MARKER_MIN_CIRC:
                            (x, y), r = cv2.minEnclosingCircle(cnt)
                            if r >= 3:
                                cands.append((x, y, a))
            if len(cands) >= 4:
                break
        except:
            pass

    if len(cands) < 4:
        return None, {"markers_found": False, "count": len(cands)}

    cands.sort(key=lambda t: t[2], reverse=True)
    pts = np.array([[t[0], t[1]] for t in cands[:14]], dtype=np.float32)

    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmin(-pts[:, 0] + pts[:, 1])]
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]

    return order_pts([tl, tr, br, bl]), {"markers_found": True, "count": len(cands)}

def warp(img, corners):
    dst = np.array([[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))

def get_roi(img, roi):
    x1n, y1n, x2n, y2n = roi
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def prep_bin(gray):
    # tăng ổn định khi ánh sáng không đều
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 35, 10)

def cell_score(bin_img, cx, cy, r):
    h, w = bin_img.shape
    cx, cy, r = int(cx), int(cy), int(r)
    roi = bin_img[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
    if roi.size == 0:
        return 0.0

    # lấy vùng “lõi” nhỏ hơn để tránh ăn viền vòng tròn in sẵn
    mask = np.zeros_like(roi)
    rr = max(3, int(r * 0.55))
    cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2), rr, 255, -1)
    tot = np.count_nonzero(mask)
    if tot == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(roi, roi, mask=mask))) / tot

def read_grid(bin_img, rows, cols):
    h, w = bin_img.shape
    r = max(6, min(22, int(min(w / (cols * 3.2), h / (rows * 3.2)))))
    scores = []
    for i in range(rows):
        row_scores = []
        for j in range(cols):
            cx = (j + 0.5) * (w / cols)
            cy = (i + 0.5) * (h / rows)
            row_scores.append(cell_score(bin_img, cx, cy, r))
        scores.append(row_scores)
    return scores

def select_bubble(scores):
    if not scores:
        return None
    best = int(np.argmax(scores))
    s = sorted(scores, reverse=True)
    top = s[0]
    second = s[1] if len(s) > 1 else 0.0
    if top >= FILL_THRESHOLD and (top - second) >= MIN_GAP:
        return best
    return None

def read_student_id(warped):
    roi_img, box = get_roi(warped, STUDENT_ID_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    binary = prep_bin(gray)

    scores = read_grid(binary, STUDENT_ROWS, STUDENT_COLS)

    digits = []
    for c in range(STUDENT_COLS):
        col_scores = [scores[r][c] for r in range(STUDENT_ROWS)]  # 0..9 từ trên xuống
        sel = select_bubble(col_scores)
        digits.append(sel)

    if any(d is None for d in digits):
        return "0", {"warning": "Không đọc được mã HS", "digits": digits, "box": box}

    sid_raw = "".join(str(d) for d in digits)  # ví dụ 011
    try:
        sid = str(int(sid_raw))  # 011 -> 11
    except:
        sid = sid_raw
    return sid, {"student_id_raw": sid_raw, "digits": digits, "box": box}

# --------- HoughCircles helpers (đọc đáp án) ----------
def kmeans_1d(vals, k, iters=25):
    vals = np.array(vals, dtype=float)
    if len(vals) == 0:
        return None, None
    # init theo quantile
    centers = np.quantile(vals, np.linspace(0, 1, k))
    for _ in range(iters):
        dist = np.abs(vals[:, None] - centers[None, :])
        labels = dist.argmin(axis=1)
        new_centers = np.array([
            vals[labels == i].mean() if np.any(labels == i) else centers[i]
            for i in range(k)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    order = np.argsort(centers)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    centers = centers[order]
    return labels, centers

def detect_circles(gray):
    g = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=18,
        param1=80, param2=18,
        minRadius=7, maxRadius=18
    )
    if circles is None:
        return np.empty((0, 3))
    return circles[0]

def fill_ratio(binary, x, y, r):
    h, w = binary.shape
    x, y, r = int(round(x)), int(round(y)), int(round(r))
    r2 = max(3, int(r * 0.55))  # lõi nhỏ hơn để tránh viền in
    x1, x2 = max(0, x - r2), min(w, x + r2)
    y1, y2 = max(0, y - r2), min(h, y + r2)
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mask = np.zeros_like(roi)
    cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2), r2 - 1, 255, -1)
    tot = np.count_nonzero(mask)
    if tot == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(roi, roi, mask=mask))) / tot

def read_answers(warped, total_q=10):
    roi_img, box = get_roi(warped, ANSWERS_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    binary = prep_bin(gray)

    circles = detect_circles(gray)
    if len(circles) < total_q * 3:
        return [None] * total_q, {"warning": "Không đủ bong bóng để đọc đáp án", "circles": int(len(circles)), "box": box}

    xs = circles[:, 0]
    ys = circles[:, 1]

    col_labels, col_centers = kmeans_1d(xs, 4)
    row_labels, row_centers = kmeans_1d(ys, total_q)

    answers = []
    picks = []

    for r in range(total_q):
        ratios = [0.0] * 4
        for c in range(4):
            idx = np.where((row_labels == r) & (col_labels == c))[0]
            if len(idx) == 0:
                ratios[c] = 0.0
                continue

            # lấy circle gần nhất tâm hàng/cột
            ii = idx[np.argmin(np.abs(ys[idx] - row_centers[r]) + 0.5 * np.abs(xs[idx] - col_centers[c]))]
            ratios[c] = fill_ratio(binary, circles[ii, 0], circles[ii, 1], circles[ii, 2])

        sel = select_bubble(ratios)
        picks.append(sel)
        answers.append(CHOICES[sel] if sel is not None else None)

    return answers, {"answers_picks": picks, "box": box, "circles": int(len(circles))}

def grade(student_ans, key, threshold):
    total = len(key)
    score = 0
    for i in range(total):
        a = (student_ans[i] or "").strip().upper()
        k = str(key[i]).strip().upper()
        if a and a == k:
            score += 1
    pct = int(round((score / total) * 100)) if total > 0 else 0
    return score, pct, "PASS" if pct >= threshold else "FAIL"

# ==================== API ====================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "QuickGrader OMR v3.1"}), 200

@app.route("/process_omr", methods=["POST"])
def process_omr():
    try:
        data = request.json or {}
        image_data = data.get("image") or data.get("image_base64")
        answer_key = data.get("answer_key", [])
        total_q = int(data.get("total_questions", len(answer_key) or 10))
        threshold = int(data.get("pass_threshold", 80))

        if not image_data:
            return jsonify({"success": False, "error": "Missing image"}), 400
        if not answer_key:
            return jsonify({"success": False, "error": "Missing answer_key"}), 400
        if total_q <= 0:
            total_q = 10

        img = decode_img(image_data)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        blur_var = check_blur(img)
        if blur_var < BLUR_THRESHOLD:
            return jsonify({
                "success": False,
                "error": "Image too blurry",
                "blur_variance": round(blur_var, 2)
            }), 422

        corners, marker_info = find_markers(img)
        if corners is None:
            return jsonify({"success": False, "error": "Cannot find 4 markers"}), 422

        warped = warp(img, corners)

        student_id, id_debug = read_student_id(warped)
        answers, ans_debug = read_answers(warped, total_q)

        score, pct, status = grade(answers, answer_key, threshold)

        return jsonify({
            "success": True,
            "student_id": str(student_id),
            "student_name": f"Hoc sinh {student_id}",
            "answers": [a if a else "" for a in answers],
            "score": score,
            "percentage": pct,
            "status": status,
            # bật debug khi cần soi lỗi (tắt cũng được)
            "debug": {
                "markers": marker_info,
                "id": id_debug,
                "answers": ans_debug
            }
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
