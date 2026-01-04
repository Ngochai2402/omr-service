from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import imutils
from imutils.perspective import four_point_transform

app = Flask(__name__)
CORS(app)

# Cấu hình vị trí tương đối (Cần điều chỉnh theo mẫu phiếu thực tế của bạn)
# Giả sử phiếu có 4 marker ở 4 góc để căn chỉnh (Warp)
# Vùng ID: Nằm ở phần trên, Vùng Đáp án: Nằm ở phần dưới

def decode_base64_image(base64_string):
    """Chuyển đổi chuỗi base64 thành ảnh OpenCV"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def process_image_logic(image, answer_key):
    """Hàm xử lý ảnh chính bằng OpenCV"""
    # 1. Tiền xử lý
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Tìm khung bài thi (dựa trên contour lớn nhất có 4 cạnh)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    doc_cnt = None

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break

    if doc_cnt is None:
        raise ValueError("Không tìm thấy khung bài thi trong ảnh.")

    # 3. Biến đổi phối cảnh để làm phẳng tờ giấy
    paper = four_point_transform(gray, doc_cnt.reshape(4, 2))
    thresh = cv2.threshold(paper, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # --- PHẦN NHẬN DIỆN MÃ HỌC SINH (Giả sử 3 chữ số đầu tiên) ---
    # Trong thực tế, bạn cần định nghĩa vùng tọa độ (ROI) chính xác của phần ID
    # Ở đây tôi lấy ví dụ chia vùng trên cùng thành 3 cột (mã HS 3 số)
    h, w = thresh.shape
    id_roi = thresh[int(h*0.05):int(h*0.25), int(w*0.6):int(w*0.95)]
    
    student_id = ""
    # Logic: Chia id_roi thành các cột và hàng để tìm ô tô đậm nhất
    # (Để đơn giản, tôi sẽ trả về giá trị giả lập nếu vùng này không chuẩn)
    # Tuy nhiên khung thuật toán là:
    # cols = np.array_split(id_roi, 3, axis=1) -> tìm ô đậm trong mỗi col
    student_id = "123" # Giá trị mặc định nếu chưa định nghĩa tọa độ ROI chính xác

    # --- PHẦN CHẤM ĐÁP ÁN ---
    # Giả sử vùng đáp án chiếm từ 30% đến 90% chiều cao tờ giấy
    ans_roi = thresh[int(h*0.3):int(h*0.9), int(w*0.1):int(w*0.9)]
    
    total_q = len(answer_key)
    detected_answers = []
    score = 0
    
    # Chia vùng đáp án thành các dòng (mỗi dòng 1 câu)
    rows = np.array_split(ans_roi, total_q, axis=0)
    
    for i, row in enumerate(rows):
        # Chia dòng thành 4 ô (A, B, C, D)
        options = np.array_split(row, 4, axis=1)
        bubbled = []
        
        for j, opt in enumerate(options):
            # Đếm số pixel trắng (đã được invert từ mực đen) trong ô
            total = cv2.countNonZero(opt)
            bubbled.append((total, j))
        
        # Sắp xếp để tìm ô đậm nhất
        bubbled.sort(key=lambda x: x[0], reverse=True)
        choice_idx = bubbled[0][1]
        choice_char = chr(65 + choice_idx) # 0->A, 1->B...
        
        detected_answers.append(choice_char)
        if choice_char == answer_key[i]:
            score += 1

    return {
        "student_id": student_id,
        "answers": detected_answers,
        "score": score,
        "total": total_q
    }

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK', 'message': 'OMR Service is active'})

@app.route('/process_omr', methods=['POST'])
def process_omr():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        answer_key = data.get('answer_key', [])
        pass_threshold = data.get('pass_threshold', 80)
        
        # Giải mã và xử lý
        image = decode_base64_image(data['image'])
        result = process_image_logic(image, answer_key)
        
        percentage = round((result['score'] / result['total']) * 100) if result['total'] > 0 else 0
        status = 'PASS' if percentage >= pass_threshold else 'FAIL'
        
        return jsonify({
            'success': True,
            'student_id': result['student_id'],
            'student_name': f"Học sinh {result['student_id']}",
            'answers': result['answers'],
            'score': result['score'],
            'percentage': percentage,
            'status': status,
            'debug': {
                'total_questions': result['total']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
