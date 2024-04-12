from cv2 import VideoCapture

class CameraFrameRecoder:
    def __init__(self, camera_url):
        """
        Initialize the CameraFrameRecoder class.

        Args:
            camera_url (str): URL or device index of the camera.
        """
        self.cap = VideoCapture(camera_url)

    def read_frame(self):
        '''
        This function is used to get frame from cap.
        It will create a generator and use key "yield" to get frame.
        '''
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # If unable to read frame, log an error and break the loop
                self.logger.log_error("Can not read frame from camera")
                break
            yield frame
        
    def release(self):
        """
        Release the camera resource.
        """
        self.cap.release()
