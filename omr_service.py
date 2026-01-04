#!/usr/bin/env python3
"""
QuickGrader OMR Service v3.0
Hệ thống chấm bài trắc nghiệm tự động - Production Ready
Tích hợp với n8n webhook workflow
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
MARKER_MIN_AREA = 0.0002
MARKER_MAX_AREA = 0.12
MARKER_MIN_CIRC = 0.38
FILL_THRESHOLD = 0.08
MIN_GAP = 0.02

# ROI theo phiếu HTML
STUDENT_ID_ROI = (0.20, 0.18, 0.80, 0.52)
ANSWERS_ROI = (0.06, 0.54, 0.94, 0.94)
STUDENT_COLS, STUDENT_ROWS = 3, 10
CHOICES = ["A", "B", "C", "D"]

# ==================== CORE FUNCTIONS ====================
def decode_img(b64):
    if ',' in b64: b64 = b64.split(',')[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

def check_blur(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def order_pts(pts):
    pts = np.array(pts, dtype="float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")

def find_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    area = gray.size
    cands = []
    
    for m in [
        lambda: cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7),
        lambda: cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        lambda: cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    ]:
        try:
            for cnt in cv2.findContours(m(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                a = cv2.contourArea(cnt)
                if area*MARKER_MIN_AREA < a < area*MARKER_MAX_AREA:
                    p = cv2.arcLength(cnt, True)
                    if p > 0 and 4*np.pi*(a/(p*p)) >= MARKER_MIN_CIRC:
                        (x,y),r = cv2.minEnclosingCircle(cnt)
                        if r >= 3: cands.append((x,y,a))
            if len(cands) >= 4: break
        except: pass
    
    if len(cands) < 4: return None, {"markers_found": False}
    cands.sort(key=lambda t: t[2], reverse=True)
    pts = np.array([[t[0],t[1]] for t in cands[:14]], dtype=np.float32)
    tl, tr, br, bl = pts[np.argmin(pts[:,0]+pts[:,1])], pts[np.argmin(-pts[:,0]+pts[:,1])], pts[np.argmax(pts[:,0]+pts[:,1])], pts[np.argmin(pts[:,0]-pts[:,1])]
    return order_pts([tl,tr,br,bl]), {"markers_found": True}

def warp(img, corners):
    dst = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(corners, dst), (WARP_W, WARP_H))

def get_roi(img, roi):
    x1n, y1n, x2n, y2n = roi
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)
    return img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)], (x1,y1,x2,y2)

def prep_bin(gray):
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)

def cell_score(bin_img, cx, cy, r):
    h, w = bin_img.shape
    cx, cy, r = int(cx), int(cy), int(r)
    roi = bin_img[max(0,cy-r):min(h,cy+r), max(0,cx-r):min(w,cx+r)]
    if roi.size == 0: return 0.0
    mask = np.zeros_like(roi)
    cv2.circle(mask, (min(r,roi.shape[1]//2), min(r,roi.shape[0]//2)), max(2,r-2), 255, -1)
    tot = np.count_nonzero(mask)
    return float(np.count_nonzero(cv2.bitwise_and(roi, roi, mask=mask)))/tot if tot > 0 else 0.0

def read_grid(bin_img, rows, cols):
    h, w = bin_img.shape
    r = max(4, min(20, int(min(w/(cols*3.2), h/(rows*3.2)))))
    return [[cell_score(bin_img, (j+0.5)*(w/cols), (i+0.5)*(h/rows), r) for j in range(cols)] for i in range(rows)]

def select_bubble(scores):
    if not scores: return None
    best_idx = int(np.argmax(scores))
    sorted_s = sorted(scores, reverse=True)
    return best_idx if sorted_s[0] >= FILL_THRESHOLD and (sorted_s[0] - (sorted_s[1] if len(sorted_s)>1 else 0)) >= MIN_GAP else None

def read_student_id(warped):
    roi_img, box = get_roi(warped, STUDENT_ID_ROI)
    binary = prep_bin(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY))
    w = binary.shape[1]
    centers = [int(w*0.25), int(w*0.50), int(w*0.75)]
    half_w = int(w*0.12)
    digits = []
    
    for c in centers:
        col = binary[:, max(0,c-half_w):min(w,c+half_w)]
        scores = read_grid(col, STUDENT_ROWS, 1)
        sel = select_bubble([scores[i][0] for i in range(STUDENT_ROWS)])
        digits.append(sel)
    
    if any(d is None for d in digits):
        return "0", {"warning": "Không đọc được mã HS"}
    
    sid = "".join(str(d) for d in digits)
    return str(int(sid)) if sid.isdigit() else sid, {"student_id_raw": sid}

def read_answers(warped, total_q):
    roi_img, box = get_roi(warped, ANSWERS_ROI)
    binary = prep_bin(cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY))
    scores = read_grid(binary, total_q, 4)
    answers = []
    
    for i in range(total_q):
        sel = select_bubble(scores[i])
        answers.append(CHOICES[sel] if sel is not None else None)
    
    return answers, {"answers_picks": [select_bubble(scores[i]) for i in range(total_q)]}

def grade(student_ans, key, threshold):
    total = len(key)
    score = sum(1 for i in range(total) if student_ans[i] and str(student_ans[i]).upper() == str(key[i]).upper())
    pct = int(round((score/total)*100)) if total > 0 else 0
    return score, pct, "PASS" if pct >= threshold else "FAIL"

# ==================== API ====================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "QuickGrader OMR v3.0"}), 200

@app.route('/process_omr', methods=['POST'])
def process_omr():
    try:
        data = request.json or {}
        image_data = data.get('image') or data.get('image_base64')
        answer_key = data.get('answer_key', [])
        total_q = int(data.get('total_questions', len(answer_key)))
        threshold = int(data.get('pass_threshold', 80))
        
        if not image_data: return jsonify({"success": False, "error": "Missing image"}), 400
        if not answer_key: return jsonify({"success": False, "error": "Missing answer_key"}), 400
        
        img = decode_img(image_data)
        if img is None: return jsonify({"success": False, "error": "Invalid image"}), 400
        
        blur_var = check_blur(img)
        if blur_var < BLUR_THRESHOLD: 
            return jsonify({"success": False, "error": "Image too blurry", "blur_variance": round(blur_var,2)}), 422
        
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
            "status": status
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
