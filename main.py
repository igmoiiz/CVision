import cv2

from src.camera import Camera


def main():
    camera = Camera()

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to capture frame.")
            break

        cv2.imshow("VisionSense", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()


if __name__ == "__main__":
    main()