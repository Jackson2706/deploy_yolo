import onnxruntime
import numpy as np
import cv2

class YOLOv8Onnx:

    def __init__(self, 
                 onnx_model_path: str, 
                 input_shape: tuple, 
                 color_padding: tuple,
                 confidence_threshold: float, 
                 nms_threshold: float,
                 label_list: dict):
        """
        Initialize YOLOv5 ONNX model.

        Args:
            onnx_model_path (str): Path to the ONNX model file.
            input_shape (tuple): Input shape of the model in the format (height, width).
            color_padding (tuple): Color for padding when resizing the image.
            confidence_threshold (float): Confidence threshold for detections.
            nms_threshold (float): Non-maximum suppression threshold for detections.
            label_list (dict): Dictionary containing class labels.
        """
        # Create an ONNX Runtime session for the model
        self.session = onnxruntime.InferenceSession(onnx_model_path)
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.color_padding = color_padding
        self.names = label_list
        
    def _preprocessing(self, frame):
        self.img_height, self.img_width = frame.shape[:2]

        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (640, 640))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor
    
    def _execute(self, input_data):
        """
        Run inference on the input data.

        Args:
            input_data: Preprocessed input data for the model.

        Returns:
            output_data: Raw output data from the model.
        """
        return self.session.run(
            None, 
            {self.session.get_inputs()[0].name: input_data})[0]

    def _postprocessing(self, output_data):
        """
        Perform postprocessing on the model output.

        Args:
            output_data: Raw output data from the model.

        Returns:
            detections: List of detections after applying confidence threshold and NMS.
        """
        detections = []
        output_data = output_data.transpose(0, 2, 1)
        grid_size = output_data.shape[1]

        x_min = output_data[0, :, 0]
        y_min = output_data[0, :, 1]
        x_max = output_data[0, :, 2]
        y_max = output_data[0, :, 3]
        class_probs = output_data[0, :, 4:]

        class_id = np.argmax(class_probs, axis=1)
        class_prob = np.max(class_probs, axis=1)
        print(class_prob)
        # Filter detections based on confidence threshold
        mask = class_prob > self.confidence_threshold
        detections = [{
            'box': [x_min[i] - (x_max[i]) / 2,
                    y_min[i] - (y_max[i]) / 2,
                    x_max[i] + x_min[i] - (x_max[i]) / 2,
                    y_max[i] + y_min[i] - (y_max[i]) / 2],
            'class_id': class_id[i],
            'confidence': class_prob[i]
        } for i in range(len(mask)) if mask[i]]

        # Apply non-maximum suppression
        if detections:
            boxes = np.array([detection['box'] for detection in detections])
            confidences = np.array([detection['confidence'] for detection in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.confidence_threshold, self.nms_threshold)
            if indices.size > 0:
                indices = indices.flatten()
                detections = [detections[i] for i in indices]
            else:
                detections = []
        return detections

    def runInference(self, frame):
        """
        Run inference on the input frame.

        Args:
            frame: Input frame (image).

        Returns:
            results: List of detections with bounding boxes and class labels.
        """
        input_data = self._preprocessing(frame=frame)
        output_data = self._execute(input_data=input_data)
        results = self._postprocessing(output_data=output_data)
        return results

    def drawbox(self, frame, results):
        """
        Draw bounding boxes and class labels on the input frame.

        Args:
            frame: Input frame (image).
            results: List of detections with bounding boxes and class labels.

        Returns:
            image_draw: Frame with bounding boxes and labels drawn.
        """
        image_draw = frame
        for value in results:
            x, y, x_max, y_max = value["box"]
            class_name = self.names[value["class_id"]]
            cv2.rectangle(image_draw, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
            cv2.putText(image_draw, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
                    thickness=2)
            
        return image_draw