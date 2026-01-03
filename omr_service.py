from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# ======================================================
# 1) CONFIG CHUNG (MẪU CỐ ĐỊNH + CHỐNG LỆCH ẢNH CHỤP)
# ======================================================
WARP_W = 900
WARP_H = 1300

# Ngưỡng mực tô (tỉ lệ pixel mực trong vòng tròn)
FILL_THRESHOLD = 0.22

# Nếu top1-top2 sát nhau -> coi là không chắc (tô mờ / tô 2 ô)
MIN_GAP = 0.06

# Check ảnh mờ: variance of Laplacian
BLUR_MIN_VAR = 80.0

# Bật debug (trả thêm scores)
DEFAULT_DEBUG = False


# ======================================================
# 2) ROI (tỉ lệ trên ảnh đã warp 900x1300)
#    Thầy có thể chỉnh nhẹ nếu cần, nhưng thường dùng được.
# ======================================================
# Vùng khung "MÃ HỌC SINH"
STUDENT_ROI = (0.22, 0.17, 0.78, 0.49)
STUDENT_COLS = 3
STUDENT_ROWS = 10  # 0..9

# Vùng khung "CÂU TRẢ LỜI"
ANS_ROI = (0.08, 0.57, 0.92, 0.93)
CHOICES = ["A", "B", "C", "D"]


# ======================================================
# 3) ROUTES
# ======================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "message": "QuickGrader OMR Service running",
        "version": "2.1.0-fixed-template-robust"
    })


@app.route("/process_omr", methods=["POST"])
def process_omr():
    try:
        data = request.json or {}

        answer_key = data.get("answer_key") or []
        pass_threshold = int(data.get("pass_threshold", 80) or 80)

        # n8n/app có thể gửi "image" hoặc "image_base64"
        image_data = data.get("image") or data.get("image_base64")
        debug = bool(data.get("debug", DEFAULT_DEBUG))

        if not image_data:
            return _err(400, "Missing image (image/image_base64)")

        if not answer_key:
            return _err(400, "answer_key is empty")

        img = _decode_base64_image(image_data)

        # 1) Check blur
        blur_var = _blur_var(img)
        if blur_var < BLUR_MIN_VAR:
            return _err(422, "Image is blurry. Please retake.", debug_payload={
                "blur_var": blur_var,
                "blur_min_var": BLUR_MIN_VAR
            })

        # 2) Warp theo 4 marker
        warped, warp_info = _warp_to_template(img)
        if warped is None:
            return _err(422, "Cannot find 4 corner markers. Please retake.", debug_payload={
                **warp_info,
                "blur_var": blur_var
            })

        total_questions = len(answer_key)

        # 3) Read student id
        student_id, stu_dbg = _read_student_id(warped)
        if not student_id:
            return _err(422, "Student ID not detected (unclear bubbles). Please retake.", debug_payload={
                "blur_var": blur_var,
                **warp_info,
                **stu_dbg
            })

        # 4) Read answers
        answers, ans_dbg = _read_answers(warped, total_questions)

        # 5) Grade
        score, percentage, status, stats = _grade(answers, answer_key, pass_threshold)

        resp = {
            "success": True,
            "student_id": str(student_id),
            "student_name": f"Hoc sinh {student_id}",
            "answers": [a if a is not None else "" for a in answers],
            "score": score,
            "percentage": percentage,
            "status": status,
        }

        if debug:
            resp["debug"] = {
                "blur_var": blur_var,
                **warp_info,
                **stu_dbg,
                **ans_dbg,
                **stats
            }

        return jsonify(resp)

    except Exception as e:
        return _err(500, str(e))


# ======================================================
# 4) CORE FUNCTIONS
# ======================================================
def _err(code: int, msg: str, debug_payload=None):
    payload = {"success": False, "error": msg}
    if debug_payload:
        payload["debug"] = debug_payload
    return jsonify(payload), code


def _decode_base64_image(data_url: str):
    if "," in data_url:
        b64 = data_url.split(",", 1)[1]
    else:
        b64 = data_url

    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


def _blur_var(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def _find_corner_markers(img_bgr):
    """
    Tìm 4 chấm đen tròn ở 4 góc.
    Trả về 4 điểm (tl,tr,br,bl) hoặc None.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 7
    )

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    img_area = w * h

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.001:
            continue
        if area > img_area * 0.06:
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue

        circularity = 4 * np.pi * (area / (peri * peri))
        if circularity < 0.55:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 8:
            continue

        candidates.append((x, y, area, circularity))

    if len(candidates) < 4:
        return None, {"markers_found": False, "marker_candidates": len(candidates)}

    # chọn top theo area
    candidates.sort(key=lambda t: t[2], reverse=True)
    top = candidates[:14]
    pts = np.array([[t[0], t[1]] for t in top], dtype=np.float32)

    tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
    tr = pts[np.argmin(-pts[:, 0] + pts[:, 1])]
    br = pts[np.argmax(pts[:, 0] + pts[:, 1])]
    bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]

    ordered = _order_points([tl, tr, br, bl])
    return ordered, {"markers_found": True, "marker_candidates": len(candidates)}


def _warp_to_template(img_bgr):
    corners, info = _find_corner_markers(img_bgr)
    if corners is None:
        return None, info

    dst = np.array([
        [0, 0],
        [WARP_W - 1, 0],
        [WARP_W - 1, WARP_H - 1],
        [0, WARP_H - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img_bgr, M, (WARP_W, WARP_H))

    info2 = {
        **info,
        "warp_size": f"{WARP_W}x{WARP_H}",
        "corners": corners.tolist(),
    }
    return warped, info2


def _roi(img_bgr, roi_norm):
    x1n, y1n, x2n, y2n = roi_norm
    h, w = img_bgr.shape[:2]
    x1 = int(x1n * w); y1 = int(y1n * h)
    x2 = int(x2n * w); y2 = int(y2n * h)
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h, y2))
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def _prep_binary(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )
    return thr


def _cell_fill_score(bin_img, cx, cy, r):
    """
    bin_img: THRESH_BINARY_INV (mực=255)
    score = tỉ lệ pixel mực trong mask vòng tròn.
    """
    h, w = bin_img.shape[:2]
    cx = int(cx); cy = int(cy); r = int(r)

    x1 = max(0, cx - r); x2 = min(w, cx + r)
    y1 = max(0, cy - r); y2 = min(h, cy + r)

    roi = bin_img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    mask = np.zeros_like(roi, dtype=np.uint8)
    # center trong ROI
    cxi = min(r, roi.shape[1] - 1)
    cyi = min(r, roi.shape[0] - 1)
    cv2.circle(mask, (cxi, cyi), max(2, r - 2), 255, -1)

    ink = cv2.bitwise_and(roi, roi, mask=mask)
    filled = int(np.count_nonzero(ink))
    total = int(np.count_nonzero(mask))
    return float(filled) / float(total) if total > 0 else 0.0


def _read_row_choice_scores(bin_img, rows, cols):
    """
    Tạo ma trận scores [rows][cols] theo lưới đều trong ROI.
    """
    h, w = bin_img.shape[:2]
    r = int(min(w / (cols * 3.2), h / (rows * 3.2)))
    r = max(6, min(16, r))

    scores = []
    for i in range(rows):
        row = []
        for j in range(cols):
            cx = (j + 0.5) * (w / cols)
            cy = (i + 0.5) * (h / rows)
            row.append(_cell_fill_score(bin_img, cx, cy, r))
        scores.append(row)
    return scores


def _pick_one(scores_row, threshold=FILL_THRESHOLD):
    """
    scores_row: list scores của các lựa chọn trong 1 câu (vd 4 lựa chọn A,B,C,D)
    return: index hoặc None
    """
    best = int(np.argmax(scores_row))
    sorted_scores = sorted(scores_row, reverse=True)
    best_score = sorted_scores[0]
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    if best_score < threshold:
        return None
    if (best_score - second) < MIN_GAP:
        return None
    return best


def _read_student_id(warped_bgr):
    roi_img, box = _roi(warped_bgr, STUDENT_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    bin_img = _prep_binary(gray)

    h, w = bin_img.shape[:2]
    col_w = w / STUDENT_COLS

    digits = []
    digit_scores = []

    for c in range(STUDENT_COLS):
        x1 = int(c * col_w)
        x2 = int((c + 1) * col_w)
        part = bin_img[:, x1:x2]  # 10 hàng, 1 cột

        # scores 10x1
        scores = _read_row_choice_scores(part, STUDENT_ROWS, 1)
        col_scores = [scores[i][0] for i in range(STUDENT_ROWS)]

        best_row = int(np.argmax(col_scores))
        sorted_scores = sorted(col_scores, reverse=True)

        if sorted_scores[0] < FILL_THRESHOLD:
            digits.append(None)
        elif (sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)) < MIN_GAP:
            digits.append(None)
        else:
            digits.append(best_row)

        digit_scores.append(col_scores)

    if any(d is None for d in digits):
        return None, {
            "student_roi": box,
            "student_digits": digits,
            "student_digit_scores": digit_scores
        }

    # ghép 3 chữ số
    sid = "".join(str(d) for d in digits)
    # chuẩn hoá bỏ số 0 đầu (nếu muốn)
    sid_norm = str(int(sid)) if sid.isdigit() else sid

    return sid_norm, {
        "student_roi": box,
        "student_digits": digits,
        "student_id_raw": sid
    }


def _read_answers(warped_bgr, total_questions):
    roi_img, box = _roi(warped_bgr, ANS_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    bin_img = _prep_binary(gray)

    # Coi ROI đáp án là grid: total_questions hàng x 4 cột (A,B,C,D)
    scores = _read_row_choice_scores(bin_img, total_questions, 4)

    answers = []
    picks = []
    for i in range(total_questions):
        idx = _pick_one(scores[i])
        picks.append(idx)
        answers.append(CHOICES[idx] if idx is not None else None)

    return answers, {
        "answers_roi": box,
        "answers_picks": picks,
        # nếu debug bật thì sẽ trả thêm scores
        "answers_scores": scores
    }


def _grade(answers, answer_key, pass_threshold):
    total = len(answer_key)
    score = 0
    detected = 0
    blank = 0
    wrong = 0

    for i in range(total):
        a = answers[i]
        if a is None or a == "":
            blank += 1
            continue
        detected += 1
        if str(a).upper() == str(answer_key[i]).upper():
            score += 1
        else:
            wrong += 1

    percentage = int(round((score / total) * 100))
    status = "PASS" if percentage >= pass_threshold else "FAIL"

    stats = {
        "total_questions": total,
        "detected_answers": detected,
        "blank_answers": blank,
        "wrong_answers": wrong,
        "pass_threshold": pass_threshold
    }

    return score, percentage, status, stats


if __name__ == "__main__":
    # Railway/Heroku style
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
