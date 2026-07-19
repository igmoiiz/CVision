import time
import cv2

from src.camera import Camera
from src.detector import ObjectDetector
from src.renderer import Renderer
from src.fps import FPS
from src.analytics import Analytics


def main():

    camera = Camera()
    detector = ObjectDetector()
    renderer = Renderer()
    fps = FPS()
    analytics = Analytics()

    while True:

        success, frame = camera.read()

        if not success:
            print("Failed to capture frame.")
            break

        # Measure inference time
        start_time = time.perf_counter()

        results = detector.detect(frame)

        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000

        # Update analytics
        analytics.update(results)

        counts = analytics.get_counts()

        # Draw detections
        frame = renderer.draw(frame, results, counts)

        # FPS
        current_fps = fps.update()

        cv2.putText(
            frame,
            f"FPS : {current_fps}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Inference Time
        cv2.putText(
            frame,
            f"Inference : {inference_time:.1f} ms",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("VisionSense", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()


if __name__ == "__main__":
    main()