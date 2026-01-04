// ===== THAY THẾ HÀM handleCaptureAndSend =====
const handleCaptureAndSend = async () => {
    try {
        if (!videoRef.current) {
            throw new Error("Camera chưa sẵn sàng");
        }

        setIsScanning(true);

        // ✅ Chụp nhiều frame, chọn frame nét nhất
        console.log('📸 Bắt đầu chụp burst frames...');
        const best = await captureBestFrameBase64(videoRef.current, {
            frames: 7,
            gapMs: 70,
            maxW: 1400,
            quality: 0.85,
            analyzeW: 320
        });

        if (!best?.dataUrl) {
            throw new Error("Không chụp được ảnh từ camera");
        }

        console.log(`✅ Đã chụp xong (sharpness score: ${best.score.toFixed(2)})`);

        // ✅ Chuẩn bị payload cho OMR Service
        const omrPayload = {
            image: best.dataUrl,
            answer_key: liveLesson.answerKey,
            total_questions: liveLesson.totalQuestions,
            pass_threshold: liveLesson.threshold
        };

        console.log('📤 Đang gửi tới OMR Service...', {
            url: OMR_SERVICE_URL,
            total_questions: omrPayload.total_questions,
            answer_key: omrPayload.answer_key
        });

        // ✅ Gọi OMR Service
        const omrResponse = await fetch(OMR_SERVICE_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(omrPayload)
        });

        if (!omrResponse.ok) {
            throw new Error(`OMR Service HTTP error: ${omrResponse.status}`);
        }

        const omrResult = await omrResponse.json();
        console.log('📊 Kết quả từ OMR Service:', omrResult);

        // ✅ Kiểm tra kết quả
        if (!omrResult.success) {
            // Hiển thị lỗi chi tiết
            const errorMessages = {
                'marker_not_found': '❌ Không tìm thấy 4 marker góc.\n\nVui lòng:\n• Chụp cả 4 góc phiếu\n• Đảm bảo marker rõ ràng\n• Không bị che khuất',
                'invalid_student_id': '❌ Không đọc được mã học sinh.\n\nVui lòng:\n• Kiểm tra học sinh đã tô đúng mã\n• Tô đậm, đủ kín\n• Chỉ tô 1 ô mỗi cột',
                'no_data': '❌ Lỗi dữ liệu gửi lên server',
                'missing_image': '❌ Thiếu ảnh',
                'missing_answer_key': '❌ Thiếu đáp án'
            };
            
            const errorMsg = errorMessages[omrResult.error] || omrResult.message || 'Lỗi không xác định';
            throw new Error(errorMsg);
        }

        // ✅ Tìm thông tin học sinh
        const student = liveStudents.find(s => s.id === String(omrResult.student_id));
        
        if (!student) {
            throw new Error(`Không tìm thấy học sinh có mã: ${omrResult.student_id}\n\nHọc sinh này có thể:\n• Không thuộc lớp này\n• Tô sai mã số`);
        }

        // ✅ Tạo kết quả
        const result = {
            studentId: String(omrResult.student_id),
            studentName: student.name,
            score: omrResult.score,
            percentage: omrResult.percentage,
            status: omrResult.status,
            answers: omrResult.answers,
            gradingDetails: omrResult.grading_details,
            scannedAt: new Date().toISOString()
        };

        console.log('✅ Kết quả chấm:', result);

        // ✅ Lưu vào Firebase
        await db.collection('artifacts')
            .doc(appId)
            .collection(`results_${liveLesson.id}`)
            .doc(result.studentId)
            .set(result);

        console.log('✅ Đã lưu vào Firebase');

        // ✅ Cập nhật history
        setHistory(prev => [result, ...prev]);
        if (historyRef.current) {
            historyRef.current.scrollTop = 0;
        }

        // ✅ Gửi webhook tới n8n (lưu MySQL)
        await sendWebhook(N8N_WEBHOOK_RESULT, {
            lesson_id: liveLesson.id,
            teacher_id: user.wp_user_id,
            student_id: result.studentId,
            student_name: result.studentName,
            score: result.score,
            total_questions: liveLesson.totalQuestions,
            percentage: result.percentage,
            status: result.status,
            answers: JSON.stringify(result.answers),
            scanned_at: result.scannedAt
        });

        console.log('✅ Đã gửi webhook tới n8n');

        // ✅ Tắt torch và đóng camera
        await tryEnableTorch(cameraStreamRef.current, false);
        closeCamera();

        // ✅ Hiển thị thông báo thành công
        const passIcon = result.status === 'PASS' ? '🎉' : '📝';
        alert(`${passIcon} Quét thành công!\n\n` +
              `Học sinh: ${student.name}\n` +
              `Điểm: ${result.score}/${liveLesson.totalQuestions} (${result.percentage}%)\n` +
              `Kết quả: ${result.status === 'PASS' ? 'ĐẠT ✅' : 'CHƯA ĐẠT ⚠️'}`);

    } catch (error) {
        console.error('❌ Lỗi khi quét:', error);
        
        // Hiển thị lỗi chi tiết
        alert(error.message || "❌ Quét thất bại. Vui lòng thử lại.");
        
    } finally {
        setIsScanning(false);
    }
};
