import time
from pathlib import Path

import matplotlib.pyplot as plt
from tactus_yolov7 import Yolov7, resize
from tactus_data import skeletonization
from tactus_data import retracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from tactus_data import skeletonization
from tactus_model.utils.tracker import FeatureTracker
from tactus_model.utils.classifier import Classifier
from tactus_live.stream import Stream,get_dartec_rtsp
from tactus_data.utils.visualisation import pipeline_visualisation
from kafka_producer import kafka_producer
import cv2





def main():
    computing_device = "cuda:0"
    model_yolov7 = Yolov7(skeletonization.MODEL_WEIGHTS_PATH, computing_device)
    deepsort = DeepSort(n_init=3, max_age=1)
    feature_tracker = FeatureTracker(deepsort, window_size=9, angles_to_compute=[])
    classifier = Classifier()
    classifier = classifier.load(Path("D:\Documents\Cranfield\GDP\TACTUS-live\\tactus_live\data\model\pickle.json"))
    rtsp_url = get_dartec_rtsp('x','x','x','x')
    producer_ip = 'x'
    topic_name = 'x'
    #producer = kafka_producer(producer_ip,topic_name)

    # Message parameters
    EventID = 1
    SensorID = "Camera3"
    Priority = 1
    VirtualInterCoord = "Not Defined" # Position on screen
    XPos = "not Defined" # Lat
    YPos = "not Defined" # Long
    print("Start Stream")
    bbx = []
    stream = Stream("D:\Documents\Cranfield\GDP\Video_for_hilda\\Dartect_Camera_G1.mp4",target_fps=10)
    fig,ax = plt.subplots()
    if not stream.isOpened():
        print("Error, cannot read stream")
        exit()
    while stream.isOpened():
        ret, frame = stream.read()
        if ret is True:
            img = resize(frame)
            skeletons = model_yolov7.predict_frame(img)
            if len(skeletons) == 0:
                continue

            feature_tracker.track_skeletons(skeletons,img)
            print(feature_tracker.rolling_windows)
            for skeleton_id, (success, features) in feature_tracker.extract():
                if not success:
                    continue
                prediction = classifier.predict([features])
                pipeline_visualisation(ax,skeletons[int(skeleton_id)-1]["keypoints"],prediction[0])
                print(skeleton_id,prediction)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            plt.pause(0.001)
            ax.clear()
            #bbx= skeleton_bbx(skeletons)

            #preprocessing = preprocessing(skeleton)
            #prediction = predict(preprocessing)
            prediction =1
            #producer.set_json_message(EventID,
            #                          prediction,
            #                          f'Event : {prediction} happened at Camera n°{SensorID}',
            #                          SensorID,
            #                          Priority,
            #                          VirtualInterCoord ,
            #                          XPos,
            #                          YPos)
            #producer.poll() # Flush pending msg
            #producer.produce()
            #producer.flush() # Flush msg
            EventID +=1
            time.sleep(0.1)


    stream.release()


main()
