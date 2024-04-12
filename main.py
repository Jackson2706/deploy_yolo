import cv2

from config import config
from streaming.streaming import CameraFrameRecoder
from model_deployment.yoloOnnxModel import YoloOnnxModel

# Khởi tạo CameraFrameRecoder và YoloOnnxModel từ các cấu hình trong config

camera = CameraFrameRecoder(camera_url=config.CAMERA_URL)
yoloDetector = YoloOnnxModel(onnx_model_path=config.YOLO_ONNX_MODEL_PATH,
                             input_shape=config.INPUT_SHAPE,
                             color_padding=config.COLOR_PADDING,
                             confidence_threshold=config.CONFIDENCE_THRESHOLD,
                             nms_threshold=config.NMS_THRESHOLD,
                             label_list=config.LABELS)

# Vòng lặp để xử lý từng frame từ camera và dự đoán bằng YOLOv5

for frame in camera.read_frame():
    # Thực hiện dự đoán trên frame hiện tại
    outputs = yoloDetector.runInference(frame)

    # Vẽ các bounding box và label lên frame
    draw_image = yoloDetector.drawbox(frame, outputs)

    # Hiển thị frame với bounding box và label
    cv2.imshow("test", draw_image)

    # Nếu nhấn phím 'q', thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên của camera sau khi kết thúc
camera.release()
