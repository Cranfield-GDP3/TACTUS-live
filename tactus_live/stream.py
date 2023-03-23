from time import time
from typing import Union
import cv2

class Stream():
    def __init__(self, stream_id: Union[str, int], desired_fps: int) -> None:
        self.stream_id = stream_id
        self._cap = cv2.VideoCapture(self.stream_id)
        self.desired_fps = desired_fps
        self.fps = None
        self.extract_freq = None

    def compute_stream_frequency(self):
        """evaluate the frame rate over a period of 5 seconds"""
        start = time()
        count = 0

        while self._cap.isOpened():
            ret, _ = self._cap.read()
            if ret is True:
                count += 1

                if time() - start > 5:
                    break

        self.fps = round(count / (time() - start), 2)
        self.extract_freq = self.fps / self.desired_fps

    _count = 0
    def read(self):
        self._count += 1

        if self._count % self.extract_freq == 0:
            return self._cap.read()

        return False, False

    def isOpened(self):
        return self._cap.isOpened()
