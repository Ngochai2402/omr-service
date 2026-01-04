#!/usr/bin/env python3
"""
Script Python đơn giản để chấm phiếu trắc nghiệm
Không cần Flask, chạy trực tiếp từ command line
"""

import cv2
import numpy as np
import sys
import os

# ==================== CẤU HÌNH ====================

# Kích thước canvas sau khi căn chỉnh
WARP_W, WARP_H = 900, 1300
MARKER_MARGIN = 60

# Ngưỡng phát hiện
MIN_FILLED_SCORE = 0.08  # Điểm tối thiểu để coi là đã tô
MIN_GAP_SCORE = 0.02     # Khoảng cách giữa câu trả lời đúng nhất và thứ 2

# Vùng ROI (x1, y1, x2, y2) - tọa độ chuẩn hóa 0-1
ID_ROI = (0.15, 0.15, 0.70, 0.45)       # Vùng ID học sinh
ANSWER_ROI = (0.10, 0.50, 0.90, 0.92)   # Vùng đáp án

# Layout phiếu
ID_COLS = 3      # 3 cột (Trăm, Chục, Đơn vị)
ID_ROWS = 10     # 10 hàng (0-9)
ANS_ROWS = 10    # 10 câu hỏi
ANS_COLS = 4     # 4 đáp án (A, B, C, D)

CHOICES = ["A", "B", "C", "D"]

# ==================== HÀM PHỤ TRỢ ====================

def tim_4_goc(img):
    """
    Tìm 4 góc đen của phiếu
    Return: 4 điểm góc [TL, TR, BR, BL] hoặc None
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện cạnh
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=2)
    
    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lọc các contour hình vuông (markers)
    markers = []
    h, w = img.shape[:2]
    min_area = (w * h) * 0.001  # Tối thiểu 0.1% diện tích ảnh
    max_area = (w * h) * 0.05   # Tối đa 5% diện tích ảnh
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Xấp xỉ hình dạng
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Kiểm tra có phải hình vuông (4 góc)
            if len(approx) == 4:
                x, y, w_rect, h_rect = cv2.boundingRect(approx)
                aspect_ratio = float(w_rect) / h_rect
                
                # Tỷ lệ gần vuông (0.8 - 1.2)
                if 0.8 < aspect_ratio < 1.2:
                    # Lấy tâm của marker
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        markers.append((cx, cy))
    
    if len(markers) < 4:
        print(f"❌ Chỉ tìm thấy {len(markers)}/4 góc!")
        return None
    
    # Sắp xếp 4 góc: TL, TR, BR, BL
    markers = sorted(markers, key=lambda p: p[1])  # Sắp xếp theo Y
    top_2 = sorted(markers[:2], key=lambda p: p[0])  # 2 góc trên, sắp xếp theo X
    bottom_2 = sorted(markers[2:4], key=lambda p: p[0])  # 2 góc dưới
    
    corners = np.array([
        top_2[0],      # Top-Left
        top_2[1],      # Top-Right
        bottom_2[1],   # Bottom-Right
        bottom_2[0]    # Bottom-Left
    ], dtype=np.float32)
    
    return corners


def can_chinh_phieu(img, corners):
    """
    Căn chỉnh phiếu về dạng thẳng
    """
    dst = np.array([
        [MARKER_MARGIN, MARKER_MARGIN],
        [WARP_W - MARKER_MARGIN, MARKER_MARGIN],
        [WARP_W - MARKER_MARGIN, WARP_H - MARKER_MARGIN],
        [MARKER_MARGIN, WARP_H - MARKER_MARGIN]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (WARP_W, WARP_H))
    
    return warped


def lay_roi(img, roi_norm):
    """
    Lấy vùng ROI từ tọa độ chuẩn hóa
    """
    x1n, y1n, x2n, y2n = roi_norm
    h, w = img.shape[:2]
    
    x1 = int(x1n * w)
    y1 = int(y1n * h)
    x2 = int(x2n * w)
    y2 = int(y2n * h)
    
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def chuan_hoa_sang(gray):
    """
    Chuẩn hóa độ sáng để giảm ảnh hưởng của bóng
    """
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    bg = np.where(bg == 0, 1, bg)
    
    normalized = cv2.divide(gray.astype(np.float32), bg.astype(np.float32))
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized.astype(np.uint8)


def tinh_diem_o_tron(binary, cx, cy, r):
    """
    Tính điểm của ô tròn (cao = đã tô đậm)
    """
    h, w = binary.shape
    cx, cy, r = int(cx), int(cy), int(r)
    
    x1 = max(0, cx - r)
    x2 = min(w, cx + r)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r)
    
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    
    # Tạo mask hình tròn
    mask = np.zeros_like(roi, dtype=np.uint8)
    roi_h, roi_w = roi.shape
    mask_cx = min(r, roi_w // 2)
    mask_cy = min(r, roi_h // 2)
    
    # Vòng tròn bên trong (ô tô)
    inner_r = max(2, int(r * 0.6))
    cv2.circle(mask, (mask_cx, mask_cy), inner_r, 255, -1)
    inner_mean = cv2.mean(roi, mask=mask)[0]
    
    # Vòng tròn bên ngoài (nền)
    ring_mask = np.zeros_like(roi, dtype=np.uint8)
    outer_r = max(inner_r + 2, int(r * 0.9))
    cv2.circle(ring_mask, (mask_cx, mask_cy), outer_r, 255, -1)
    cv2.circle(ring_mask, (mask_cx, mask_cy), inner_r, 0, -1)
    
    if cv2.countNonZero(ring_mask) > 0:
        bg_mean = cv2.mean(roi, mask=ring_mask)[0]
    else:
        bg_mean = 255
    
    # Điểm = (nền - trong) / 255 (càng cao = tô càng đậm)
    score = (bg_mean - inner_mean) / 255.0
    return max(0.0, score)


def doc_luoi_diem(binary, rows, cols):
    """
    Đọc điểm của tất cả ô trong lưới
    """
    h, w = binary.shape
    cell_w = w / cols
    cell_h = h / rows
    r = int(min(cell_w, cell_h) * 0.3)
    r = max(5, min(30, r))
    
    scores = []
    for i in range(rows):
        row_scores = []
        for j in range(cols):
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h
            score = tinh_diem_o_tron(binary, cx, cy, r)
            row_scores.append(score)
        scores.append(row_scores)
    
    return scores


def chon_o_tron(scores_row):
    """
    Chọn ô tròn được tô trong 1 hàng
    Return: index của ô được chọn hoặc None
    """
    if not scores_row:
        return None
    
    sorted_scores = sorted(enumerate(scores_row), key=lambda x: x[1], reverse=True)
    
    best_idx, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    
    # Kiểm tra ngưỡng
    if best_score < MIN_FILLED_SCORE:
        return None
    
    gap = best_score - second_score
    if gap < MIN_GAP_SCORE:
        return None  # Không rõ ràng (tô 2 ô hoặc tô mờ)
    
    return best_idx


def doc_ma_so_hoc_sinh(warped):
    """
    Đọc mã số học sinh (3 cột x 10 hàng)
    """
    roi_img, box = lay_roi(warped, ID_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Chuẩn hóa sáng
    normalized = chuan_hoa_sang(gray)
    
    # Chuyển sang nhị phân
    binary = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    
    # Đọc điểm
    scores = doc_luoi_diem(binary, ID_ROWS, ID_COLS)
    
    # Chọn số cho mỗi cột
    digits = []
    for col in range(ID_COLS):
        col_scores = [scores[row][col] for row in range(ID_ROWS)]
        selected = chon_o_tron(col_scores)
        digits.append(selected if selected is not None else 0)
    
    # Ghép thành mã số
    ma_so = "".join(str(d) for d in digits)
    ma_so_int = int(ma_so)
    
    return str(ma_so_int), digits, scores


def doc_dap_an(warped, so_cau):
    """
    Đọc đáp án (N câu x 4 cột ABCD)
    """
    roi_img, box = lay_roi(warped, ANSWER_ROI)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # Chuẩn hóa sáng
    normalized = chuan_hoa_sang(gray)
    
    # Chuyển sang nhị phân
    binary = cv2.adaptiveThreshold(
        normalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    
    # Đọc điểm
    scores = doc_luoi_diem(binary, so_cau, ANS_COLS)
    
    # Chọn đáp án cho mỗi câu
    answers = []
    picks = []
    
    for i in range(so_cau):
        selected = chon_o_tron(scores[i])
        picks.append(selected)
        
        if selected is None:
            answers.append("")
        else:
            answers.append(CHOICES[selected])
    
    return answers, picks, scores


def cham_diem(dap_an_hoc_sinh, dap_an_dung, nguong_dat):
    """
    Chấm điểm
    """
    tong_cau = len(dap_an_dung)
    diem = 0
    
    for i in range(tong_cau):
        if dap_an_hoc_sinh[i] and dap_an_hoc_sinh[i].upper() == dap_an_dung[i].upper():
            diem += 1
    
    phan_tram = int(round((diem / tong_cau) * 100)) if tong_cau > 0 else 0
    trang_thai = "ĐẠT" if phan_tram >= nguong_dat else "CHƯA ĐẠT"
    
    return diem, phan_tram, trang_thai


def ve_ket_qua(warped, id_digits, answer_picks):
    """
    Vẽ kết quả lên ảnh để debug
    """
    result_img = warped.copy()
    
    # Vẽ ROI ID
    roi_img, (x1, y1, x2, y2) = lay_roi(warped, ID_ROI)
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(result_img, "ID", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Vẽ các ô được chọn cho ID
    roi_h = y2 - y1
    roi_w = x2 - x1
    cell_w = roi_w / ID_COLS
    cell_h = roi_h / ID_ROWS
    
    for col, digit in enumerate(id_digits):
        if digit is not None:
            cx = x1 + int((col + 0.5) * cell_w)
            cy = y1 + int((digit + 0.5) * cell_h)
            cv2.circle(result_img, (cx, cy), 12, (0, 0, 255), 3)
    
    # Vẽ ROI đáp án
    roi_img, (x1, y1, x2, y2) = lay_roi(warped, ANSWER_ROI)
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(result_img, "DAP AN", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Vẽ các ô được chọn cho đáp án
    roi_h = y2 - y1
    roi_w = x2 - x1
    cell_w = roi_w / ANS_COLS
    cell_h = roi_h / ANS_ROWS
    
    for row, pick in enumerate(answer_picks):
        if pick is not None:
            cx = x1 + int((pick + 0.5) * cell_w)
            cy = y1 + int((row + 0.5) * cell_h)
            cv2.circle(result_img, (cx, cy), 12, (0, 0, 255), 3)
    
    return result_img


# ==================== HÀM CHÍNH ====================

def cham_phieu(duong_dan_anh, dap_an_dung, nguong_dat=80, luu_ket_qua=True):
    """
    Hàm chính để chấm phiếu
    
    Args:
        duong_dan_anh: Đường dẫn đến ảnh phiếu
        dap_an_dung: List đáp án đúng, VD: ["A","B","C","D","A","B","C","D","A","B"]
        nguong_dat: Phần trăm để đạt (mặc định 80%)
        luu_ket_qua: Có lưu ảnh kết quả không
    
    Returns:
        Dictionary chứa kết quả
    """
    
    print("="*60)
    print("🎓 BẮT ĐẦU CHẤM PHIẾU")
    print("="*60)
    
    # 1. Đọc ảnh
    print("📸 Đọc ảnh:", duong_dan_anh)
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print("❌ Không đọc được ảnh!")
        return None
    
    print(f"✅ Kích thước ảnh: {img.shape[1]}x{img.shape[0]}")
    
    # 2. Tìm 4 góc
    print("🔍 Tìm 4 góc marker...")
    corners = tim_4_goc(img)
    if corners is None:
        print("❌ Không tìm thấy đủ 4 góc!")
        return None
    
    print("✅ Đã tìm thấy 4 góc")
    
    # 3. Căn chỉnh phiếu
    print("📐 Căn chỉnh phiếu...")
    warped = can_chinh_phieu(img, corners)
    print("✅ Đã căn chỉnh")
    
    # 4. Đọc mã số học sinh
    print("🔢 Đọc mã số học sinh...")
    ma_so, id_digits, id_scores = doc_ma_so_hoc_sinh(warped)
    print(f"✅ Mã số: {ma_so}")
    
    # 5. Đọc đáp án
    print("📝 Đọc đáp án...")
    dap_an, answer_picks, ans_scores = doc_dap_an(warped, len(dap_an_dung))
    print(f"✅ Đáp án: {dap_an}")
    
    # 6. Chấm điểm
    print("📊 Chấm điểm...")
    diem, phan_tram, trang_thai = cham_diem(dap_an, dap_an_dung, nguong_dat)
    
    # 7. Hiển thị kết quả
    print("\n" + "="*60)
    print("📋 KẾT QUẢ CHẤM PHIẾU")
    print("="*60)
    print(f"👤 Mã số học sinh: {ma_so}")
    print(f"📝 Đáp án học sinh: {' '.join(dap_an)}")
    print(f"✅ Đáp án đúng:     {' '.join(dap_an_dung)}")
    print(f"📊 Điểm: {diem}/{len(dap_an_dung)}")
    print(f"📈 Phần trăm: {phan_tram}%")
    print(f"🎯 Kết quả: {trang_thai}")
    print("="*60)
    
    # 8. Lưu ảnh kết quả
    if luu_ket_qua:
        result_img = ve_ket_qua(warped, id_digits, answer_picks)
        output_path = duong_dan_anh.replace(".", "_result.")
        cv2.imwrite(output_path, result_img)
        print(f"💾 Đã lưu ảnh kết quả: {output_path}")
    
    # 9. Trả về kết quả
    return {
        "ma_so": ma_so,
        "dap_an": dap_an,
        "diem": diem,
        "phan_tram": phan_tram,
        "trang_thai": trang_thai,
        "id_digits": id_digits,
        "answer_picks": answer_picks
    }


# ==================== CHẠY THỬ ====================

if __name__ == "__main__":
    # Kiểm tra tham số
    if len(sys.argv) < 2:
        print("Cách sử dụng:")
        print("  python cham_phieu_don_gian.py <đường_dẫn_ảnh>")
        print("\nVí dụ:")
        print("  python cham_phieu_don_gian.py phieu_hoc_sinh_1.jpg")
        sys.exit(1)
    
    duong_dan_anh = sys.argv[1]
    
    # Đáp án mẫu (thay đổi theo đề của bạn)
    dap_an_dung = ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"]
    
    # Chấm phiếu
    ket_qua = cham_phieu(duong_dan_anh, dap_an_dung, nguong_dat=80, luu_ket_qua=True)
    
    if ket_qua:
        print("\n✅ Chấm phiếu thành công!")
    else:
        print("\n❌ Chấm phiếu thất bại!")
