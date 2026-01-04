# omr_service.py
import base64
import binascii
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12MB

# =========================
# Helpers
# =========================

def decode_image_base64(image_base64: str) -> np.ndarray:
    """Accept raw base64 or dataURL: data:image/jpeg;base64,... -> return BGR image."""
    if not image_base64 or not isinstance(image_base64, str):
        raise ValueError("image_base64 missing or not a string")

    s = image_base64.strip()
    if s.startswith("data:"):
        if "," not in s:
            raise ValueError("Invalid data URL (missing comma)")
        s = s.split(",", 1)[1].strip()

    s = "".join(s.split())
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64: {e}")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed (not a valid image bytes)")
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def find_corner_markers(img_bgr: np.ndarray) -> np.ndarray:
    """
    Find 4 black circular-ish markers near corners.
    Return 4 points (tl,tr,br,bl) in original image coordinates.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # tăng tương phản nhẹ
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold đảo để marker đen -> trắng (dễ đếm)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # làm sạch
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (w * h) * 0.0002:  # lọc nhiễu nhỏ
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 10 or ch < 10:
            continue

        # độ tròn tương đối (circularity)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4 * math.pi * area / (peri * peri)

        # marker tròn: circularity ~ 0.6-1.0 (nới lỏng để chịu ảnh chụp lệch)
        if circularity < 0.45:
            continue

        # tâm
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        candidates.append((area, cx, cy))

    if len(candidates) < 4:
        raise ValueError("Không tìm đủ 4 marker góc. Hãy chụp rõ 4 góc và marker đen.")

    # lấy top marker theo diện tích
    candidates.sort(key=lambda t: t[0], reverse=True)
    top = candidates[:12]  # dự phòng có logo/đốm lớn
    pts = np.array([[c[1], c[2]] for c in top], dtype=np.float32)

    # chọn 4 điểm gần 4 góc nhất:
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    chosen = []
    used = set()
    for ci in range(4):
        cx, cy = corners[ci]
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        # chọn điểm gần nhất góc, chưa dùng
        order = np.argsort(dists)
        pick = None
        for idx in order:
            if idx not in used:
                pick = idx
                break
        if pick is None:
            raise ValueError("Marker góc bị trùng/không phân biệt được.")
        used.add(pick)
        chosen.append(pts[pick])

    chosen = np.array(chosen, dtype=np.float32)
    rect = order_points(chosen)  # tl,tr,br,bl
    return rect


def warp_to_a4(img_bgr: np.ndarray, rect: np.ndarray, out_w: int = 1654, out_h: int = 2339) -> np.ndarray:
    """
    Warp theo 4 marker về khung A4 chuẩn.
    Mặc định 1654x2339 ~ A4 @200dpi-ish (đủ cho OMR).
    """
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
    return warped


def binarize_for_marking(warped_bgr: np.ndarray) -> np.ndarray:
    """Return binary image where filled marks are 1 (white) after invert."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # tô đen -> trắng
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return th


@dataclass
class GridSpec:
    x: float  # left ratio 0..1
    y: float  # top ratio
    w: float  # width ratio
    h: float  # height ratio
    rows: int
    cols: int


def cell_rect(spec: GridSpec, r: int, c: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = int(spec.x * W)
    y0 = int(spec.y * H)
    ww = int(spec.w * W)
    hh = int(spec.h * H)
    cw = ww / spec.cols
    ch = hh / spec.rows
    x = int(x0 + c * cw)
    y = int(y0 + r * ch)
    return x, y, int(cw), int(ch)


def mark_score(th_inv: np.ndarray, x: int, y: int, w: int, h: int, pad: int = 3) -> float:
    """Compute fill ratio inside cell (higher = more filled)."""
    H, W = th_inv.shape[:2]
    x1 = max(0, x + pad)
    y1 = max(0, y + pad)
    x2 = min(W, x + w - pad)
    y2 = min(H, y + h - pad)
    roi = th_inv[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    # th_inv: filled mark ~ white(255)
    return float(np.count_nonzero(roi)) / float(roi.size)


def read_one_choice_per_row(th_inv: np.ndarray, spec: GridSpec, choices: List[str], min_fill: float = 0.18) -> Tuple[List[Optional[str]], List[Dict[str, Any]]]:
    """
    Each row: select max-filled among cols.
    Return answers + debug per row.
    """
    H, W = th_inv.shape[:2]
    out = []
    dbg = []
    for r in range(spec.rows):
        scores = []
        for c in range(spec.cols):
            x, y, cw, ch = cell_rect(spec, r, c, W, H)
            s = mark_score(th_inv, x, y, cw, ch)
            scores.append(s)
        best = int(np.argmax(scores))
        best_s = scores[best]
        # kiểm tra đậm đủ
        if best_s < min_fill:
            out.append(None)
            status = "blank"
        else:
            # nếu 2 ô cùng đậm (tô 2 đáp án)
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2 and (sorted_scores[0] - sorted_scores[1]) < 0.03 and sorted_scores[1] >= min_fill:
                out.append(None)
                status = "multi"
            else:
                out.append(choices[best])
                status = "ok"
        dbg.append({"row": r, "scores": [round(x, 3) for x in scores], "status": status})
    return out, dbg


# =========================
# Layout Specs (PHÙ HỢP mẫu HTML bên dưới)
# =========================
# A4 warped size: 1654 x 2339
# - Mã học sinh: 10 cột (0..9), mỗi cột 10 hàng (digit 0..9)
# - Đáp án: 40 câu, 4 lựa chọn A,B,C,D

STUDENT_ID_SPEC = GridSpec(
    x=0.12, y=0.15, w=0.76, h=0.20,
    rows=10, cols=10
)
# Trong spec này: rows = 10 (digit 0..9), cols = 10 (vị trí chữ số 1..10)
# => ta đọc theo COL: mỗi cột chọn 1 hàng -> digit

ANSWER_SPEC = GridSpec(
    x=0.12, y=0.40, w=0.76, h=0.50,
    rows=40, cols=4
)

DIGITS = [str(i) for i in range(10)]
ABCD = ["A", "B", "C", "D"]


def read_student_id(th_inv: np.ndarray, spec: GridSpec, min_fill: float = 0.18) -> Tuple[str, Dict[str, Any]]:
    """
    spec rows=10 digits, cols=10 positions
    Read each column -> choose one row => digit.
    """
    H, W = th_inv.shape[:2]
    digits = []
    debug_cols = []
    for c in range(spec.cols):
        scores = []
        for r in range(spec.rows):
            x, y, cw, ch = cell_rect(spec, r, c, W, H)
            s = mark_score(th_inv, x, y, cw, ch)
            scores.append(s)
        best_r = int(np.argmax(scores))
        best_s = scores[best_r]

        if best_s < min_fill:
            digits.append("?")
            status = "blank"
        else:
            sorted_scores = sorted(scores, reverse=True)
            if len(sorted_scores) >= 2 and (sorted_scores[0] - sorted_scores[1]) < 0.03 and sorted_scores[1] >= min_fill:
                digits.append("?")
                status = "multi"
            else:
                digits.append(str(best_r))
                status = "ok"

        debug_cols.append({"col": c, "scores": [round(x, 3) for x in scores], "status": status})

    return "".join(digits), {"min_fill": min_fill, "cols": debug_cols}


# =========================
# API
# =========================

@app.post("/process_omr")
def process_omr():
    try:
        data = request.get_json(force=True, silent=False)
        if not data:
            return jsonify({"ok": False, "error": "Empty JSON body"}), 400

        image_base64 = data.get("image_base64") or data.get("image")
        if not image_base64:
            return jsonify({"ok": False, "error": "Missing image_base64 (or image)"}), 400

        answer_key = data.get("answer_key")  # optional
        pass_threshold = float(data.get("pass_threshold", 90))

        img = decode_image_base64(image_base64)

        rect = find_corner_markers(img)
        warped = warp_to_a4(img, rect, out_w=1654, out_h=2339)
        th_inv = binarize_for_marking(warped)

        student_id, sid_debug = read_student_id(th_inv, STUDENT_ID_SPEC, min_fill=0.18)

        answers, ans_debug = read_one_choice_per_row(th_inv, ANSWER_SPEC, ABCD, min_fill=0.18)

        # scoring if answer_key provided
        score = None
        correct = None
        passed = None
        if isinstance(answer_key, list) and len(answer_key) == 40:
            correct = 0
            for i in range(40):
                if answers[i] is not None and str(answers[i]).upper() == str(answer_key[i]).upper():
                    correct += 1
            score = round(correct * 100.0 / 40.0, 2)
            passed = score >= pass_threshold

        return jsonify({
            "ok": True,
            "student_id": student_id,
            "answers": answers,
            "score": score,
            "correct": correct,
            "passed": passed,
            "debug": {
                "student_id": sid_debug,
                "answers": ans_debug,
                "note": "Nếu hay bị '?': tăng min_fill hoặc in đậm vòng tròn hơn / chụp sáng hơn."
            }
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.get("/health")
def health():
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    # local run: python omr_service.py
    app.run(host="0.0.0.0", port=8080, debug=True)
