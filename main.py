import cv2
import threading
from streaming.streaming import CameraFrameRecoder
from model_deployment.yoloOnnxModel import YoloOnnxModel
from model_deployment.yolov8Onnx import YOLOv8Onnx
# from firebase_manager import FirebaseManager
import time
from config import config

def main():
    camera = CameraFrameRecoder(camera_url=config.CAMERA_URL)
    yoloDetector = YOLOv8Onnx(onnx_model_path=config.YOLO_ONNX_MODEL_PATH,
                                 input_shape=config.INPUT_SHAPE,
                                 color_padding=config.COLOR_PADDING,
                                 confidence_threshold=config.CONFIDENCE_THRESHOLD,
                                 nms_threshold=config.NMS_THRESHOLD,
                                 label_list=config.LABELS)
    # firebase_manager = FirebaseManager(cred_path=config.CERTIFICATE,
    #                                    database_url=config.DATABASE_URL)
    start_time = 0
    elapsed_time = 0

    for frame in camera.read_frame():
        outputs = yoloDetector.runInference(frame)
        print(outputs)
        draw_image = yoloDetector.drawbox(frame, outputs)
        # if "drowsy" in [config.LABELS[det["class_id"]] for det in outputs]:
        #     print(1)
        #     if start_time == 0:
        #         start_time = time.time()
        #     else:
        #         elapsed_time = time.time() - start_time
        #         if elapsed_time >= 2:
        #             start_time1 = elapsed_time
        #             firebase_thread = threading.Thread(target=firebase_manager.send_data_to_firebase,
        #                                                args=(start_time1,))
        #             firebase_thread.start()
        # else:
        #     start_time = 0
        #     elapsed_time = 0
        #     firebase_thread1 = threading.Thread(target=firebase_manager.send_data_to_firebase1)
        #     firebase_thread1.start()

        cv2.imshow("test", draw_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()

if __name__ == "__main__":
    main()
