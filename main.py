import cv2 # type: ignore

from src.camera import Camera
from src.detector import ObjectDetector


def main():
    camera = Camera()
    detector = ObjectDetector()

    while True:
        success, frame = camera.read()

        if not success:
            break

        results = detector.detect(frame)

        # Draw detections on the frame
        annotated_frame = results[0].plot()

        cv2.imshow("VisionSense", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()


if __name__ == "__main__":
    main()