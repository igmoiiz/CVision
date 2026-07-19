import time


class FPS:

    def __init__(self):

        self.previous_time = time.time()

        self.fps = 0

    def update(self):

        current_time = time.time()

        elapsed = current_time - self.previous_time

        self.previous_time = current_time

        if elapsed > 0:
            self.fps = 1 / elapsed

        return int(self.fps)