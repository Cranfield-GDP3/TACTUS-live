import copy
from typing import List
from pathlib import Path
from cProfile import Profile
import tqdm

import numpy as np
import cv2
from tactus_data import PosePredictionYolov8, VideoCapture, Skeleton
from tactus_data import retracker, visualisation
from tactus_model import FeatureTracker, Classifier, PredTracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track

from tactus_live.utils.kafka_producer import KafkaProducer


def main(device: str = "cuda:0"):
    deepsort_tracker = DeepSort(n_init=5, max_age=10, device=device)
    pose_model = PosePredictionYolov8(Path("data/raw/models"), "yolov8m-pose.pt", device)

    classifier = Classifier.load(Path("data/models/BEST_RUS.pickle"))
    feature_tracker = FeatureTracker(classifier.window_size, classifier.angle_to_compute)
    pred_tracker = PredTracker()

    cap = VideoCapture(r"C:\Users\marco\Downloads\img_7350 low bitrate.mp4",
                       target_fps=10,
                       tqdm_progressbar=tqdm.tqdm())

    kafka_producer = KafkaProducer(stream_ip="1.1.1.1", topic_name="camera1/detections", sensor_id="camera1", client_id="x")

    pr1 = Profile()
    pr1.enable()
    try:
        while (cap_frame := cap.read())[1] is not None:
            frame_id, frame = cap_frame

            skeletons = pose_model.predict(frame)

            tracks: List[Track] = retracker.deepsort_reid(deepsort_tracker, frame, skeletons)
            tracks_to_del = copy.deepcopy(deepsort_tracker.tracker.del_tracks_ids)

            for track_id in tracks_to_del:
                feature_tracker.delete_track_id(track_id)
                pred_tracker.delete_track_id(track_id)

            for track in tracks:
                track_id = track.track_id

                # if the track has no new information, don't classify the action
                if track.time_since_update > 0:
                    x_left, y_top, x_right, y_bottom = track.to_ltrb()
                    new_bbox_lbrt = (x_left, y_bottom, x_right, y_top)
                    feature_tracker.duplicate_last_entry(track_id, new_bbox_lbrt)
                else:
                    skeleton = track.others
                    feature_tracker.update_rolling_window(track_id, skeleton)

                    if track.is_confirmed() and feature_tracker[track_id].is_complete():
                        features = feature_tracker[track_id].get_features()

                        prediction = classifier.predict_label([features])[0]

                        if prediction != "neutral":
                            skeleton = feature_tracker.rolling_windows[track_id].skeleton
                            pred_tracker.add_pred(track_id, prediction, skeleton)

                            kafka_producer.send_event("violence", "violence is happening on this camera")

                frame = plot_skeleton(frame, skeleton, track, feature_tracker, pred_tracker)

            save_dir = Path("data/visualisation")
            save_path = save_dir / (str(frame_id) + ".jpg")
            cv2.imwrite(str(save_path), frame)

    finally:
        pr1.disable()
        pr1.print_stats('cumtime')
        cap.release()


def plot_skeleton(frame: np.ndarray, skeleton: Skeleton, track: Track, feature_tracker: FeatureTracker, pred_tracker: PredTracker):
    track_id = track.track_id

    bbox_thickness = 2
    if track.time_since_update > 0:
        bbox_thickness = 1

    label = None
    bbox_color = visualisation.GREEN
    if not track.is_confirmed() or not feature_tracker[track_id].is_complete():
        bbox_color = visualisation.GREY
    elif track_id in pred_tracker:
        label = pred_tracker[track_id]["current_label"]
        bbox_color = visualisation.RED

    frame = visualisation.plot_bbox(frame, skeleton,
                                    color=bbox_color, thickness=bbox_thickness,
                                    label=label)
    if not feature_tracker[track_id].is_duplicated():
        frame = visualisation.plot_joints(frame, skeleton, thickness=2)

    return frame


main()
