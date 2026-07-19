import cv2

from src.camera import Camera
from src.detector import ObjectDetector
from src.renderer import Renderer


def main():

    camera = Camera()

    detector = ObjectDetector()

    renderer = Renderer()

    while True:

        success, frame = camera.read()

        if not success:
            break

        results = detector.detect(frame)

        frame = renderer.draw(frame, results)

        cv2.imshow("VisionSense", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()


if __name__ == "__main__":
    main()