from tactus_yolov7 import Yolov7, resize
from tactus_data import skeletonization
from tactus_data import retracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from tactus_live.stream import Stream


def main(computing_device: str):
    model_yolov7 = Yolov7(skeletonization.MODEL_WEIGHTS_PATH, computing_device)
    deepsort_tracker = DeepSort(n_init=3, max_age=5)

    stream = Stream(0, target_fps=10)
    while stream.isOpened():
        ret, frame = stream.read()
        if ret is True:
            img = resize(frame)
            skeletons = model_yolov7.predict_frame(img)
            skeletons = retracker.deepsort_track_frame(deepsort_tracker, img, skeletons)

    stream.release()
