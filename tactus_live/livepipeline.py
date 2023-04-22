from pathlib import Path
from tactus_yolov7 import Yolov7, resize
from deep_sort_realtime.deepsort_tracker import DeepSort
from tactus_data import skeletonization
from tactus_model.utils.tracker import FeatureTracker, PredTracker
from tactus_model.utils.classifier import Classifier
from tactus_live.stream import Stream, get_rtsp
from tactus_data.utils.visualisation import cv2_plot, plot_predict, plot_bbx
from kafka_producer import kafka_producer
import cv2

AVAILABLE_CLASSES = ['kicking', 'punching', 'pushing', 'neutral']


def init_camera_live_pipeline(model_path: Path, rstp_url: str,
                              computing_device: str = "cuda:0", flag_save: bool = False):
    """
    Initialise one live pipeline for one camera.

    Parameters
    ----------
    model_path : Path,
        Path where the model pickle file is located
    rstp_url : str,
        string of the rstp_url
    computing_device : str,
        name of the computing devices (GPU/CPU) the models will run on
    flag_save : bool,
        flag to enable saving the video to .avi format, change to True
        to enable
    """
    # Init models
    model_yolov7 = Yolov7(skeletonization.MODEL_WEIGHTS_PATH, computing_device)
    deepsort = DeepSort(n_init=3, max_age=0)
    prediction_tracker = PredTracker()
    classifier = Classifier()
    classifier = classifier.load(model_path)
    feature_tracker = FeatureTracker(deepsort, window_size=classifier.window_size, angles_to_compute=classifier.angle_to_compute)
    # Init Kafka
    producer_ip = 'x'
    topic_name = 'x'
    # producer = kafka_producer(producer_ip,topic_name) #add get kafka data function from secret file

    # Message parameters
    EventID = 1
    SensorID = "Camera1"
    Priority = 1
    VirtualInterCoord = "Not Defined"  # Position on screen
    XPos = "not Defined"  # Lat
    YPos = "not Defined"  # Long
    print("Start Stream")
    bbx = []
    first = True

    stream = Stream(rstp_url, target_fps=classifier.fps) # change direct rstp to a function loading rstp data from secret file
    if not stream.isOpened():
        print("Error, cannot read stream")
        exit()
    while stream.isOpened():
        ret, frame = stream.read()
        if ret is True:
            img = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
            img = resize(img)
            if flag_save and first:
                writer = cv2.VideoWriter_fourcc(*'XVID')
                output = cv2.VideoWriter('video_output.avi', writer, 10, (img.shape[1], img.shape[0]))
                first = False
            skeletons = model_yolov7.predict_frame(img)
            if len(skeletons) == 0:
                continue
            feature_tracker.track_skeletons(skeletons, img)
            print(feature_tracker.rolling_windows)
            count = 0
            for track_id, (success, features) in feature_tracker.extract():
                if track_id not in prediction_tracker:

                    prediction_tracker.init_track_id(track_id)
                if not success:
                    cv2_plot(img, skeletons[count],prediction_tracker[track_id])
                    continue
                print([features].shape())
                prediction = classifier.predict([features])
                if prediction[0] != 3:
                    # producer.set_json_message(EventID,
                    #                          prediction,
                    #                          f'Event : {prediction} happened at Camera n°{SensorID}',
                    #                          SensorID,
                    #                          Priority,
                    #                          VirtualInterCoord ,
                    #                          XPos,
                    #                          YPos)
                    # producer.poll() # Flush pending msg
                    # producer.produce()
                    # producer.flush() # Flush msg
                    # time.sleep(0.1)
                    EventID += 1
                prediction_tracker.add_pred(track_id, AVAILABLE_CLASSES[prediction[0]], skeletons[count]["box"])
                cv2_plot(img, skeletons[count], prediction_tracker[track_id],track_id)
                count += 1
                print(track_id, prediction)
            if flag_save:
                output.write(img)
            cv2.imshow('Dartec Camera', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    stream.release()
    if flag_save:
        output.release()
    cv2.destroyAllWindows()

