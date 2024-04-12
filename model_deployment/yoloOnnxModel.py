import onnxruntime
import cv2
import numpy as np

class YoloOnnxModel:
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
 
    def _letterbox(self, frame):
        """
        Resize and pad the input frame to match the model's input shape.

        Args:
            frame: Input frame (image).

        Returns:
            canvas: Resized and padded image.
        """
        h, w = frame.shape[:2]
        target_h, target_w = self.input_shape

        # Calculate aspect ratio
        aspect_ratio = min(target_w / w, target_h / h)

        # Calculate new width and height
        new_w = int(w * aspect_ratio)
        new_h = int(h * aspect_ratio)

        # Resize image
        resized_im = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a blank canvas of the target size
        canvas = np.full((target_h, target_w, 3), self.color_padding, dtype=np.uint8)

        # Calculate coordinates to paste resized image in the center
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # Paste the resized image onto the canvas
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_im

        return canvas
    
    def _preprocessing(self, frame):
        """
        Preprocess the input frame for model inference.

        Args:
            frame: Input frame (image).

        Returns:
            input_data: Preprocessed input data for the model.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = self._letterbox(frame=frame)
        image_resized = np.transpose(image_resized, (2, 0, 1))  # Change channel order
        image_resized = np.ascontiguousarray(image_resized)
        if len(image_resized.shape) == 3:
            image_resized = image_resized[None]  # Expand for batch dim
        input_data = image_resized.astype(np.float32)
        input_data /= 255  # Normalize
        return input_data

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
        grid_size = output_data.shape[1]

        x_min = output_data[0, :, 0]
        y_min = output_data[0, :, 1]
        x_max = output_data[0, :, 2]
        y_max = output_data[0, :, 3]
        confidence = output_data[0, :, 4]
        class_probs = output_data[0, :, 5:]

        class_id = np.argmax(class_probs, axis=1)
        class_prob = np.max(class_probs * confidence[:, np.newaxis], axis=1)

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
        image_draw = self._letterbox(frame=frame)
        for value in results:
            x, y, x_max, y_max = value["box"]
            class_name = self.names[value["class_id"]]
            cv2.rectangle(image_draw, (int(x), int(y)), (int(x_max), int(y_max)), (0, 255, 0), 1)
            cv2.putText(image_draw, class_name, (int(x), int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255],
                    thickness=2)
            
        return image_draw
