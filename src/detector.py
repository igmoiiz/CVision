from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="models/yolov8n.pt"):

        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            verbose=False
        )

        return results