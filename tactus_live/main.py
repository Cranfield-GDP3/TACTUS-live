from time import time
from tactus_yolov7 import Yolov7, resize
from tactus_data import skeletonization
from tactus_data import retracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from tactus_live.stream import Stream


def main(device: str):
    model_yolov7 = Yolov7(skeletonization.MODEL_WEIGHTS_PATH, device)
    deepsort_tracker = DeepSort(n_init=3, max_age=5)

    stream = Stream(0, 30)
    count = 0
    while stream.isOpened():
        ret, frame = stream.read()
        if ret is True:
            if count == 0:
                start = time()
            if time() - start > 15:
                break
            img = resize(frame)
            skeletons = model_yolov7.predict_frame(img)
            skeletons = retracker.deepsort_track_frame(deepsort_tracker, img, skeletons)
            count += 1
            if skeletons:
                print("found for frame ", count)

    stream.release()
    print(count/15)
