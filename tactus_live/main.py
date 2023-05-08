import copy
from typing import List
from pathlib import Path

import cv2
from tactus_data import PosePredictionYolov8, VideoCapture
from tactus_data import retracker, visualisation
from tactus_model import FeatureTracker, Classifier, PredTracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track


def main(device: str = "cuda:0"):
    # Must return a list of Skeleton with at least their bounding box
    deepsort_tracker = DeepSort(n_init=5, max_age=10)
    # Must return a list of Skeleton
    pose_model = PosePredictionYolov8(Path("data/raw/models"), "yolov8s-pose.pt", device)

    classifier = Classifier.load(Path("data/models/22.pickle"))
    feature_tracker = FeatureTracker(classifier.window_size, classifier.angle_to_compute)
    pred_tracker = PredTracker()

    cap = VideoCapture(r"C:\Users\marco\Downloads\img_7350 low bitrate.mp4", target_fps=10)

    try:
        while (cap_frame := cap.read()) is not None:
            frame_id, frame = cap_frame
            frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5)

            skeletons = pose_model.predict(frame)

            tracks: List[Track] = retracker.deepsort_reid(deepsort_tracker, frame, skeletons)
            tracks_to_del = copy.deepcopy(deepsort_tracker.tracker.del_tracks_ids)

            for track_id in tracks_to_del:
                feature_tracker.delete_track_id(track_id)
                pred_tracker.delete_track_id(track_id)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                # if there is no skeleton update, don't classify the action
                if track.time_since_update > 0:
                    x_left, y_top, x_right, y_bottom = track.to_ltrb()
                    new_bbox_lbrt = (x_left, y_bottom, x_right, y_top)
                    feature_tracker.duplicate_last_entry(track.track_id, new_bbox_lbrt)
                    continue

                skeleton = track.others

                feature_tracker.update_rolling_window(track.track_id, skeleton)

                # success, features = feature_tracker[track.track_id].get_features()

                # if success:
                    # prediction = classifier.predict([features])

                    # if prediction != "neutral":
                        # skeleton = feature_tracker.rolling_windows[track.track_id].skeleton
                        # pred_tracker.add_pred(track.track_id, prediction, skeleton)

            for track_id in feature_tracker.rolling_windows:
                skeleton = feature_tracker[track_id].skeleton

                color = (0, 255, 0)  # blue
                # if the person was violent, track it with red
                if track_id in pred_tracker:
                    color = (0, 0, 255)  # red

                if feature_tracker[track_id].is_duplicated():
                    thickness = 1
                else:
                    thickness = 2

                frame = visualisation.plot_bbox(frame, skeleton, color=color, thickness=thickness, label=track_id)
                if not feature_tracker[track_id].is_duplicated():
                    frame = visualisation.plot_joints(frame, skeleton, thickness=1)

            save_dir = Path("data/visualisation")
            save_path = save_dir / (str(frame_id) + ".jpg")
            cv2.imwrite(str(save_path), frame)

    finally:
        cap.release()


main()
