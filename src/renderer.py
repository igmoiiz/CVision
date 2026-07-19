import cv2


class Renderer:

    def __init__(self):

        self.colors = {
            "person": (0, 255, 0),
            "cell phone": (255, 0, 255),
            "bottle": (255, 255, 0),
            "laptop": (255, 0, 0),
            "chair": (0, 165, 255)
        }

        self.default_color = (0, 0, 255)

    def draw(self, frame, results):

        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])
            label = names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = self.colors.get(label, self.default_color)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text = f"{label} {confidence:.2f}"

            cv2.putText(
                frame,
                f"Objects : {len(boxes)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        return frame
    
    