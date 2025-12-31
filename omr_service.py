from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'OK',
        'message': 'QuickGrader OMR Service is running',
        'version': '1.0.0'
    })

@app.route('/process_omr', methods=['POST'])
def process_omr():
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing request data'
            }), 400
        
        answer_key = data.get('answer_key', [])
        pass_threshold = data.get('pass_threshold', 80)
        
        student_ids = ['1', '8', '9', '10', '11', '12']
        student_id = random.choice(student_ids)
        
        total_questions = len(answer_key)
        score = random.randint(0, total_questions)
        percentage = round((score / total_questions) * 100) if total_questions > 0 else 0
        status = 'PASS' if percentage >= pass_threshold else 'FAIL'
        
        choices = ['A', 'B', 'C', 'D']
        answers = [random.choice(choices) for _ in range(total_questions)]
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'student_name': 'Hoc sinh ' + student_id,
            'answers': answers,
            'score': score,
            'percentage': percentage,
            'status': status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
