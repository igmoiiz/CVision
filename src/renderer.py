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

        displayed_objects = 0

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            displayed_objects += 1

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

            if box.id is not None:
                object_id = int(box.id[0])
                text = f"{label} #{object_id} {confidence:.2f}"
            else:
                text = f"{label} {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Draw object count once, outside the loop
        cv2.putText(
            frame,
            f"Objects : {displayed_objects}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        return frame