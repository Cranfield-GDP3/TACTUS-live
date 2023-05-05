from typing import Tuple
import numpy as np


def crop_frame(frame: np.ndarray, bbox_ltwh: Tuple[float, float, float, float]) -> np.ndarray:
    """crop a bounding box out of a frame."""
    x_left, y_top, width, height = (int(x) for x in bbox_ltwh)
    subframe = frame[y_top:(y_top + height)][x_left:(x_left + width)]

    return subframe
