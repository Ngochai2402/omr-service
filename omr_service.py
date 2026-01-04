#!/usr/bin/env python3
"""
QuickGrader OMR – TNMaker-style (Refactored Single File)
Author: ChatGPT (for Hai)
"""

# =====================================================
# MODULE 1 – IMPORTS & CONFIG
# =====================================================
import cv2, base64, os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

W, H = 900, 1300
CHOICES = ["A", "B", "C", "D"]

ROI_ID     = (0.08, 0.22, 0.42, 0.56)
ROI_ANSWER = (0.05, 0.60, 0.72, 0.92)

FILL_TH = 0.12
GAP_TH  = 0.04

# =====================================================
# MODULE 2 – IMAGE IO & NORMALIZATION (TNMaker CORE)
# =====================================================
def b64_to_img(b64):
    if "," in b64:
        b64 = b64.split(",")[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def normalize_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    return clahe.apply(gray)

def adaptive_bin(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )

# =====================================================
# MODULE 3 – MARKER DETECTION & WARP
# =====================================================
def order_pts(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)]
    ], dtype="float32")

def find_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,51,7)

    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []

    for c in cnts:
        a = cv2.contourArea(c)
        if a < 300 or a > 20000: continue
        p = cv2.arcLength(c, True)
        if p == 0: continue
        circ = 4*np.pi*a/(p*p)
        if circ > 0.4:
            (x,y),_ = cv2.minEnclosingCircle(c)
            pts.append([x,y])

    if len(pts) < 4:
        return None

    return order_pts(np.array(pts[:4]))

def warp(img, corners):
    dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (W,H))

def crop(img, roi):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = int(roi[0]*w),int(roi[1]*h),int(roi[2]*w),int(roi[3]*h)
    return img[y1:y2, x1:x2]

# =====================================================
# MODULE 4 – BUBBLE DETECTION (NO GRID)
# =====================================================
def detect_circles(gray):
    return cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=18,
        param1=80, param2=18,
        minRadius=7, maxRadius=18
    )

def fill_ratio(bin_img, x, y, r):
    r = int(r * 0.55)
    roi = bin_img[int(y-r):int(y+r), int(x-r):int(x+r)]
    if roi.size == 0:
        return 0.0
    return np.count_nonzero(roi) / roi.size

def kmeans_1d(vals, k):
    vals = np.array(vals, dtype=np.float32).reshape(-1,1)
    _, labels, centers = cv2.kmeans(
        vals, k, None,
        (cv2.TERM_CRITERIA_EPS, 10, 1.0),
        10, cv2.KMEANS_PP_CENTERS
    )
    order = np.argsort(centers.flatten())
    remap = {old:new for new,old in enumerate(order)}
    return np.array([remap[l[0]] for l in labels])

# =====================================================
# MODULE 5 – READ ID / READ ANSWERS
# =====================================================
def read_student_id(warped):
    img = crop(warped, ROI_ID)
    gray = normalize_gray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    bin_img = adaptive_bin(gray)

    circles = detect_circles(gray)
    if circles is None:
        return "0"

    c = circles[0]
    rows = kmeans_1d(c[:,1], 10)
    cols = kmeans_1d(c[:,0], 3)

    digits = []
    for col in range(3):
        scores = [0]*10
        for i in range(len(c)):
            if cols[i] == col:
                scores[rows[i]] = fill_ratio(bin_img, c[i][0], c[i][1], c[i][2])
        sel = np.argmax(scores)
        if scores[sel] < FILL_TH:
            return "0"
        digits.append(str(sel))

    return str(int("".join(digits)))

def read_answers(warped):
    img = crop(warped, ROI_ANSWER)
    gray = normalize_gray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    bin_img = adaptive_bin(gray)

    circles = detect_circles(gray)
    if circles is None:
        return [None]*10

    c = circles[0]
    rows = kmeans_1d(c[:,1], 10)
    cols = kmeans_1d(c[:,0], 4)

    answers = []
    for q in range(10):
        ratios = [0]*4
        for opt in range(4):
            for i in range(len(c)):
                if rows[i]==q and cols[i]==opt:
                    ratios[opt] = fill_ratio(bin_img, c[i][0], c[i][1], c[i][2])
        best = np.argmax(ratios)
        s = sorted(ratios, reverse=True)
        if s[0] > FILL_TH and s[0]-s[1] > GAP_TH:
            answers.append(CHOICES[best])
        else:
            answers.append(None)
    return answers

# =====================================================
# MODULE 6 – API
# =====================================================
@app.route("/process_omr", methods=["POST"])
def process_omr():
    data = request.json
    img = b64_to_img(data["image"])
    answer_key = data["answer_key"]

    corners = find_markers(img)
    if corners is None:
        return jsonify({"success": False, "error": "marker_not_found"})

    warped = warp(img, corners)

    sid = read_student_id(warped)
    answers = read_answers(warped)

    score = sum(1 for i in range(10) if answers[i] == answer_key[i])
    pct = score * 10

    return jsonify({
        "success": True,
        "student_id": sid,
        "answers": answers,
        "score": score,
        "percentage": pct,
        "status": "PASS" if pct >= 80 else "FAIL"
    })

if __name__ == "__main__":
    app.run("0.0.0.0", 8000)
