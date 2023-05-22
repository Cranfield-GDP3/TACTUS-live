import copy
from typing import List
from pathlib import Path
import tqdm

import numpy as np
import cv2
from tactus_data import PosePredictionYolov8, VideoCapture, Skeleton
from tactus_data import retracker, visualisation
from tactus_model import FeatureTracker, Classifier, PredTracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track

from tactus_live.utils.kafka_producer import KafkaProducer


def main(pose_model_path: Path = Path("data/models/yolov8m-pose.pt"),
         classifier_path: Path = Path("data/models/classifier_best.pickle"),
         output_frame_dir: Path = Path("data/visualisation"),
         device: str = "cuda:0",
         kafka_producer: KafkaProducer = None):
    """
    run the entire tactus pipeline.

    Parameters
    ----------
    pose_model_path : Path, optional
        path to the yolov8 pose estimation model. Its name must remain
        unchanged, by default Path("data/models/yolov8m-pose.pt")
    classifier_path : Path, optional
        path to the classifier. A pickle file may be use to inject
        undesire code and can be a security threat,
        by default Path("data/models/classifier_best.pickle")
    output_frame_dir : Path, optional
        where to save the frames. If None, the visualisation is
        disabled, by default Path("data/visualisation")
    device : _type_, optional
        the device on which to run yolo and deepsort models,
        by default "cuda:0"
    kafka_producer : KafkaProducer, optional
        the kafka producer instance to use. If None, kafka usage
        is disabled, by default None
    """
    deepsort_tracker = DeepSort(n_init=5, max_age=10, device=device)
    pose_model = PosePredictionYolov8(pose_model_path.parent, pose_model_path.name, device)

    classifier = Classifier.load(classifier_path)
    feature_tracker = FeatureTracker(classifier.window_size, classifier.angles_to_compute)
    pred_tracker = PredTracker()

    cap = VideoCapture(r"data/videos/seq4_lq.mp4",
                       target_fps=10,
                       tqdm_progressbar=tqdm.tqdm())

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
                    skeleton = feature_tracker.duplicate_last_entry(track_id, new_bbox_lbrt)
                else:
                    skeleton = track.others
                    feature_tracker.update_rolling_window(track_id, skeleton)

                    if track.is_confirmed() and feature_tracker[track_id].is_complete():
                        features = feature_tracker[track_id].get_features()

                        prediction = classifier.predict_label([features])[0]
                        pred_tracker.add_pred(track_id, prediction)

                        if kafka_producer is not None:
                            kafka_producer.send_event("violence", "violence is happening on this camera")

                frame = plot_skeleton(frame, skeleton, track, feature_tracker, pred_tracker)

            if output_frame_dir is not None:
                save_path = output_frame_dir / (str(frame_id) + ".jpg")
                cv2.imwrite(str(save_path), frame)

    finally:
        cap.release()


def plot_skeleton(frame: np.ndarray,
                  skeleton: Skeleton,
                  track: Track,
                  feature_tracker: FeatureTracker,
                  pred_tracker: PredTracker) -> np.ndarray:
    """
    plot skeletons and their bounding box on a frame.

    Parameters
    ----------
    frame : np.ndarray
        the representation of the frame
    skeleton : Skeleton
        the Skeleton object containing just a bounding box, or a bounding
        box and keypoints.
    track : Track
        the Track object from DeepSORT
    feature_tracker : FeatureTracker
        the feature tracker. It is used to check if the rolling window
        of a skeleton is full (hence rendy to be classified), and to see if the last entry was duplicated or not.
    pred_tracker : PredTracker
        check if the skeleton made a violent action at one point prior
        to the current frame.

    Returns
    -------
    np.ndarray
        frame with the skeleton plotted on it.
    """
    track_id = track.track_id

    bbox_thickness = 4
    if track.time_since_update > 0:
        bbox_thickness = 2

    label = None
    bbox_color = visualisation.GREEN
    if not track.is_confirmed() or not feature_tracker[track_id].is_complete():
        bbox_color = visualisation.GREY
    else:
        label = pred_tracker.get_last_pred(track_id)
        if pred_tracker[track_id]["violent"]:
            bbox_color = visualisation.RED

    frame = visualisation.plot_bbox(frame, skeleton,
                                    color=bbox_color, thickness=bbox_thickness,
                                    label=label)
    if not feature_tracker[track_id].is_duplicated():
        frame = visualisation.plot_joints(frame, skeleton, thickness=2)

    return frame


if __name__ == "__main__":
    main()
