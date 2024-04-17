YOLO_ONNX_MODEL_PATH = "weights/Yolov8/best.onnx"
INPUT_SHAPE = (640, 640)
COLOR_PADDING = (114,114,114)
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
# LABELS = {
#   0: "normal",
#   1: "drowsy",
#   2: "drowsy#2",
#   3: "yawning",
#  }
LABELS = {0: 'microsleep', 1: 'neutral', 2: 'yawning'}




CAMERA_URL = 0



DATABASE_URL = "https://fir-mq3-default-rtdb.firebaseio.com"
CERTIFICATE = "google-services.json"