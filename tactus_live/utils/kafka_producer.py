import json
import logging

from confluent_kafka import Producer


class KafkaProducer(Producer):
    """extension of Producer class to send json messages more easily.

    Example
    -------
    ```
    kafka_producer = KafkaProducer(ip_adress="1.1.1.1",
                                       topic_name="camera1/detections",
                                       sensor_id="camera1",
                                       client_id="x")
    ```
    """
    def __init__(self, ip_adress: str, topic_name: str, sensor_id: str, client_id: str, log_success: bool = False):
        super().__init__({'bootstrap.servers': ip_adress,
                          'client_id': client_id})
        self.topic_name = topic_name
        self.sensor_id = sensor_id
        self.event_id = 0
        self.log_success = log_success

    def callback_report(self, err, msg):
        if err is not None:
            logging.Logger(f"Message delivery failed: {err}", logging.ERROR)
        elif self.log_success:
            topic = msg.topic()
            value = msg.value().decode('utf-8')
            logging.Logger(f"Message delivered to {topic} [{value}]")

    def send_event(self, alarm_type: str, description: str,
                   *,
                   event_id: int = None,
                   priority: int = 1,
                   VirtualInterCoord: str = "",
                   XPos: int = "",
                   YPos: int = ""):
        if event_id is None:
            event_id = self.event_id
            self.event_id += 1

        json_message = {
            "EventId": event_id,
            "AlarmType": alarm_type,
            "Description": description,
            "SensorId": self.sensor_id,
            "Priority": priority,
            "VirtualInterceptCoordinates": VirtualInterCoord,
            "XPos": XPos,
            "YPos": YPos
        }

        self.poll(timeout=0)
        self.produce(topic=self.topic_name,
                     value=json.dumps(json_message).encode('utf-8'),
                     callback=self.callback_report)
        self.flush()
