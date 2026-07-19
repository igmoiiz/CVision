import cv2 # type: ignore

class Camera:
    def __init__(self, camera_index=0):
        self.camera = cv2.VideoCapture(camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Unable to access the webcam.")

        # Camera settings
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def read(self):
        success, frame = self.camera.read()
        return success, frame

    def release(self):
        self.camera.release()
        cv2.destroyAllWindows()