from time import time
from typing import Union
import cv2

class Stream():
    def __init__(self, stream_id: Union[str, int], desired_fps: int = None) -> None:
        self.stream_id = stream_id
        self._cap = cv2.VideoCapture(self.stream_id)

        self.desired_fps = desired_fps
        self.fps = None
        self.extract_freq = None
        self.compute_stream_frequency()

    def compute_stream_frequency(self):
        """evaluate the frame rate over a period of 5 seconds"""
        count = 0

        while self.isOpened():
            ret, _ = self._cap.read()
            if ret is True:
                if count == 0:
                    start = time()

                count += 1

                if time() - start > 5:
                    break

        if count == 0:
            raise FileNotFoundError("stream probably not found")

        self.fps = round(count / (time() - start), 2)

        if self.desired_fps is None:
            self.extract_freq = 1
        else:
            self.extract_freq = int(self.fps / self.desired_fps)

            if self.extract_freq == 0:
                raise ValueError("desired_fps is higher than the stream fps")

    _count = 0
    def read(self):
        """read the sub-sampled stream according to the value of
        desired_fps"""
        ret, frame = self._cap.read()
        if ret is True:
            self._count += 1

            if self._count == self.extract_freq:
                self._count = 0
                return ret, frame

        return False, False

    def isOpened(self):
        """check if the stream is opened"""
        return self._cap.isOpened()

    def release(self):
        self._cap.release()
