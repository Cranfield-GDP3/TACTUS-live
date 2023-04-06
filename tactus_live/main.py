import time

from tactus_yolov7 import Yolov7, resize
from tactus_data import skeletonization
from tactus_data import retracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from tactus_live.stream import Stream,get_dartec_rtsp
from kafka_producer import kafka_producer
import cv2





def main(computing_device: str):
    model_yolov7 = Yolov7(skeletonization.MODEL_WEIGHTS_PATH, computing_device)
    deepsort_tracker = DeepSort(n_init=3, max_age=5)
    rtsp_url = get_dartec_rtsp('safeoper','hnTe-$k2x-!sZq-mWo9','trc.dartec.cranfield.ac.uk','9092')
    producer_ip = 'trc.dartec.cranfield.ac.uk:9092'
    topic_name = 'hildasafe01'
    producer = kafka_producer(producer_ip,topic_name)

    # Message parameters
    EventID = 1
    SensorID = "Camera3"
    Priority = 1
    VirtualInterCoord = "Not Defined" # Position on screen
    XPos = "not Defined" # Lat
    YPos = "not Defined" # Long

    stream = cv2.VideoCapture(rtsp_url)
    if not stream.isOpened():
        print("Error, cannot read stream")
        exit()
    while stream.isOpened():
        ret, frame = stream.read()
        if ret is True:
            img = resize(frame)
            skeletons = model_yolov7.predict_frame(img)
            skeletons = retracker.deepsort_track_frame(deepsort_tracker, img, skeletons)
            #preprocessing = preprocessing(skeleton)
            #prediction = predict(preprocessing)
            producer.set_json_message(EventID,
                                      prediction,
                                      f'Event : {prediction} happened at Camera n°{SensorID}',
                                      SensorID,
                                      Priority,
                                      VirtualInterCoord ,
                                      XPos,
                                      YPos)
            producer.poll() # Flush pending msg
            producer.produce()
            producer.flush() # Flush msg
            EventID +=1
            time.sleep(0.1)


    stream.release()
