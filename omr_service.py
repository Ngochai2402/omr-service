#!/usr/bin/env python3
"""
QuickGrader OMR Service v3.1 (Fix ổn định theo code bạn đang dùng)
- Flask + OpenCV
- Tìm 4 markers -> warp 900x1300
- Đọc mã HS 3 chữ số (0-9)
- Đọc 10 câu A/B/C/D
- Debug optional: trả warped/thresh base64 để kiểm tra ROI
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

# Marker detection theo tỉ lệ diện tích ảnh gốc
MARKER_MIN_AREA = 0.0002
MARKER_MAX_AREA = 0.12
MARKER_MIN_CIRC = 0.38

# Bubble selection
FILL_THRESHOLD = 0.08
MIN_GAP = 0.02

# ROI theo phiếu HTML (GIỮ NGUYÊN như code bạn đang dùng)
# Format: (x1, y1, x2, y2) theo tỉ lệ 0..1 trên ảnh WARP
STUDENT_ID_ROI = (0.20, 0.18, 0.80, 0.52)
ANSWERS_ROI    = (0.06, 0.54, 0.94, 0.94)

STUDENT_COLS, STUDENT_ROWS = 3, 10
CHOICES = ["A", "B", "C", "D"]

# ==================== TIỆN ÍCH OPENCV ====================
def find_contours(binary_img):
    """
    OpenCV 3/4: cv2.findContours trả về khác nhau.
    Hàm này đảm bảo luôn lấy được cnts.
    """
    out = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV4: (cnts, hier) ; OpenCV3: (img, cnts, hier)
    cnts = out[0] if len(out) == 2 else out[1]
    return cnts

def decode_img(b64):
    if not b64:
        return None
    if ',' in b64:
        b64 = b64.split(',', 1)[1]
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

def img_to_b64jpg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

def check_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")

def warp(img, corners):
    dst = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (WARP_W, WARP_H))

def get_roi(img, roi):
    x1n, y1n, x2n, y2n = roi
    h, w = img.shape[:2]
    x1, y1 = int(x1n*w), int(y1n*h)
    x2, y2 = int(x2n*w), int(y2n*h)
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w,   x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h,   y2))
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0], (x1,y1,x2,y2)
    return img[y1:y2, x1:x2], (x1,y1,x2,y2)

def prep_bin(gray):
    g = cv2.GaussianBlur(gray, (5,5), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 7)

# ==================== MARKER DETECTION (ỔN ĐỊNH HƠN) ====================
def circularity(area, peri):
    if peri <= 1e-6:
        return 0.0
    return float(4.0*np.pi*(area/(peri*peri)))

def find_markers(img):
    """
    Trả về (corners4, debug_info)
    corners4: ndarray shape (4,2) TL,TR,BR,BL
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    H, W = gray.shape[:2]
    img_area = float(H * W)

    # 3 kiểu threshold để tăng khả năng bắt marker
    thresh_list = [
        cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 51, 7),
        cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 31, 5),
    ]

    candidates = []
    for th in thresh_list:
        try:
            cnts = find_contours(th)
            for cnt in cnts:
                a = cv2.contourArea(cnt)
                if not (img_area*MARKER_MIN_AREA < a < img_area*MARKER_MAX_AREA):
                    continue
                p = cv2.arcLength(cnt, True)
                if p <= 0:
                    continue
                circ = circularity(a, p)
                if circ < MARKER_MIN_CIRC:
                    continue
                (x, y), r = cv2.minEnclosingCircle(cnt)
                if r < 3:
                    continue
                candidates.append((float(a), float(x), float(y), float(r)))
            if len(candidates) >= 6:
                break
        except:
            pass

    if len(candidates) < 4:
        return None, {"markers_found": False, "count": len(candidates)}

    # Sort theo area giảm dần
    candidates.sort(key=lambda t: t[0], reverse=True)

    # Lọc bớt điểm quá gần nhau (tránh 2 contour cùng 1 marker)
    picked = []
    min_dist = max(W, H) * 0.08  # khoảng cách tối thiểu giữa 2 marker
    for a, x, y, r in candidates:
        ok = True
        for _, px, py, _ in picked:
            if (x-px)**2 + (y-py)**2 < (min_dist**2):
                ok = False
                break
        if ok:
            picked.append((a, x, y, r))
        if len(picked) >= 8:
            break

    if len(picked) < 4:
        return None, {"markers_found": False, "count": len(picked), "note": "too_close_filtered"}

    # Lấy các điểm (x,y) và suy ra 4 góc
    pts = np.array([[p[1], p[2]] for p in picked], dtype=np.float32)

    # Chọn TL/TR/BR/BL chuẩn theo sum/diff
    corners = order_points(pts)

    return corners, {"markers_found": True, "candidates": len(candidates), "picked": len(picked),
                     "corners": corners.tolist()}

# ==================== OMR GRID ====================
def cell_score(bin_img, cx, cy, r):
    h, w = bin_img.shape
    cx, cy, r = int(cx), int(cy), int(r)
    y1, y2 = max(0, cy-r), min(h, cy+r)
    x1, x2 = max(0, cx-r), min(w, cx+r)
    roi = bin_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    mask = np.zeros_like(roi)
    # tâm mask trong ROI
    mx, my = roi.shape[1]//2, roi.shape[0]//2
    rad = max(2, min(mx, my, r-2))
    cv2.circle(mask, (mx, my), rad, 255, -1)
    tot = int(np.count_nonzero(mask))
    if tot <= 0:
        return 0.0
    filled = int(np.count_nonzero(cv2.bitwise_and(roi, roi, mask=mask)))
    return float(filled) / float(tot)

def read_grid(bin_img, rows, cols):
    h, w = bin_img.shape
    r = max(4, min(20, int(min(w/(cols*3.2), h/(rows*3.2)))))
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
        return None, {"reason": "empty"}
    arr = np.array(scores, dtype=np.float32)
    best = int(np.argmax(arr))
    top = float(arr[best])
    arr2 = arr.copy()
    arr2[best] = -1.0
    second = float(np.max(arr2)) if len(arr2) > 1 else 0.0

    if top < FILL_THRESHOLD:
        return None, {"reason": "too_light", "top": top, "second": second}
    if (top - second) < MIN_GAP:
        return None, {"reason": "ambiguous", "top": top, "second": second}
    return best, {"reason": "ok", "top": top, "second": second}

# ==================== READ STUDENT ID / ANSWERS ====================
def read_student_id(warped):
    roi_img, box = get_roi(warped, STUDENT_ID_ROI)
    if roi_img.size == 0:
        return "0", {"warning": "Student ROI empty", "box": box}

    binary = prep_bin(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY))
    w = binary.shape[1]

    # Giữ đúng logic bạn đang dùng (3 cột ở 25%,50%,75%)
    centers = [int(w*0.25), int(w*0.50), int(w*0.75)]
    half_w = int(w*0.12)

    digits = []
    meta_cols = []

    for c in centers:
        col = binary[:, max(0, c-half_w):min(w, c+half_w)]
        scores = read_grid(col, STUDENT_ROWS, 1)  # 10x1
        col_scores = [scores[i][0] for i in range(STUDENT_ROWS)]
        sel, m = select_bubble(col_scores)
        digits.append(sel)
        meta_cols.append({"scores": col_scores, "pick": sel, "meta": m})

    if any(d is None for d in digits):
        return "0", {"warning": "Không đọc được mã HS", "cols": meta_cols}

    sid_raw = "".join(str(d) for d in digits)
    # bỏ số 0 đầu
    try:
        sid = str(int(sid_raw))
    except:
        sid = sid_raw

    return sid, {"student_id_raw": sid_raw, "cols": meta_cols}

def read_answers(warped, total_q):
    roi_img, box = get_roi(warped, ANSWERS_ROI)
    if roi_img.size == 0:
        return [None]*total_q, {"warning": "Answers ROI empty", "box": box}

    binary = prep_bin(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY))
    scores = read_grid(binary, total_q, 4)

    answers = []
    picks = []
    metas = []

    for i in range(total_q):
        sel, m = select_bubble(scores[i])
        picks.append(sel)
        metas.append({"q": i+1, "scores": scores[i], "pick": sel, "meta": m})
        answers.append(CHOICES[sel] if sel is not None else None)

    return answers, {"answers_picks": picks, "detail": metas}

def grade(student_ans, key, threshold):
    total = len(key)
    score = 0
    for i in range(total):
        if student_ans[i] and key[i] and str(student_ans[i]).upper() == str(key[i]).upper():
            score += 1
    pct = int(round((score/total)*100)) if total > 0 else 0
    return score, pct, ("PASS" if pct >= threshold else "FAIL")

# ==================== API ====================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "QuickGrader OMR v3.1"}), 200

@app.route('/process_omr', methods=['POST'])
def process_omr():
    try:
        data = request.json or {}
        image_data = data.get('image') or data.get('image_base64')
        answer_key = data.get('answer_key', [])
        threshold = int(data.get('pass_threshold', 80))
        debug = bool(data.get('debug', False))

        # ép đúng 10 câu theo yêu cầu
        total_q = int(data.get('total_questions', 10))
        if total_q <= 0:
            total_q = 10

        if not image_data:
            return jsonify({"success": False, "error": "Missing image"}), 400
        if not answer_key:
            return jsonify({"success": False, "error": "Missing answer_key"}), 400

        # chuẩn hóa answer_key về list 10 phần tử
        if isinstance(answer_key, str):
            answer_key = [x.strip().upper() for x in answer_key.split(",") if x.strip()]
        answer_key = [str(x).upper() for x in answer_key][:total_q]
        if len(answer_key) < total_q:
            answer_key += [None] * (total_q - len(answer_key))

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
            return jsonify({
                "success": False,
                "error": "Cannot find 4 markers",
                "marker_info": marker_info
            }), 422

        warped = warp(img, corners)

        student_id, id_debug = read_student_id(warped)
        answers, ans_debug = read_answers(warped, total_q)
        score, pct, status = grade(answers, answer_key, threshold)

        resp = {
            "success": True,
            "student_id": str(student_id),
            "student_name": f"Hoc sinh {student_id}",
            "answers": [a if a else "" for a in answers],
            "score": score,
            "percentage": pct,
            "status": status
        }

        if debug:
            # trả thêm ảnh để bạn nhìn đúng/sai ROI ngay trong n8n
            gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            th_w = prep_bin(gray_w)

            # vẽ khung ROI lên warped để dễ kiểm tra
            dbg = warped.copy()
            for roi, color in [(STUDENT_ID_ROI, (0,255,0)), (ANSWERS_ROI, (255,0,0))]:
                _, (x1,y1,x2,y2) = get_roi(dbg, roi)
                cv2.rectangle(dbg, (x1,y1), (x2,y2), color, 3)

            resp["debug"] = {
                "marker_info": marker_info,
                "id_debug": id_debug,
                "ans_debug": ans_debug,
                "warped_with_roi_base64": img_to_b64jpg(dbg, 80),
                "warped_base64": img_to_b64jpg(warped, 80),
                "thresh_base64": img_to_b64jpg(th_w, 80),
                "blur_variance": round(blur_var, 2)
            }

        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
