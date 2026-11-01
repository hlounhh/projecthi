# IMOU Fall Detection - Web Presentation

Trang báo cáo & trình diễn kết quả phát hiện ngã đột ngột bằng mô hình Ensemble (YOLO11 Pose + PyTorch Posture Classifier).

## Cấu trúc thư mục
- `index.html`: Cấu trúc nội dung chính của báo cáo
- `style.css`: Toàn bộ CSS phong cách Elegant Minimalism (Responsive Desktop 100% full width & Mobile app swipe)
- `app.js`: Script IntersectionObserver tự động phát video mượt mà khi cuộn tới
- `videos/`: 4 video kết quả kiểm thử thực tế được trích xuất từ mô hình AI:
  - `result_video.mp4`: Clip 01 - Ngã chúi người phía trước
  - `result_video2.mp4`: Clip 02 - Ngã ngửa ra sau sàn nhà
  - `result_video3.mp4`: Clip 03 - Đang ngồi đứng dậy rồi ngã
  - `result_video4.mp4`: Clip 04 - Ngã nghiêng người sang một bên
