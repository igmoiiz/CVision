from collections import Counter


class Analytics:

    def __init__(self):
        self.class_counts = Counter()

    def update(self, results):

        self.class_counts.clear()

        boxes = results[0].boxes
        names = results[0].names

        for box in boxes:

            confidence = float(box.conf[0])

            if confidence < 0.50:
                continue

            class_id = int(box.cls[0])
            class_name = names[class_id]

            self.class_counts[class_name] += 1

    def get_counts(self):
        return dict(self.class_counts)

    def total_objects(self):
        return sum(self.class_counts.values())