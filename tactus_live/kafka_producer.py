from confluent_kafka import Producer
import json
import time


class kafka_producer():


    def __init__(self, ip, topic_name):
        self.ip = ip
        self.producer = Producer({'bootstrap.servers':ip,
                                  'client_id' : 'GDP3'
                                  })
        self.topic_name = topic_name
        self.json_message = None

    def callback_report(err, msg):
        if err is not None:
            print('Message delivery failed: {}'.format(err))
        else:
            print('Message delivered to {} [{}]'.format(msg.topic(), msg.value().decode('utf-8')))

    def set_json_message(self, EventId, AlarmType, Description,
                         SensorID,Priority,VirtualInterCoord,
                         XPos,YPos):
        self.json_message = {
                            "EventId": EventId,
                            "AlarmType": AlarmType,
                            "Description": Description,
                            "SensorId": SensorID,
                            "Priority": Priority,
                            "VirtualInterceptCoordinates": VirtualInterCoord,
                            "XPos": XPos,
                            "YPos": YPos
        }

    def flush(self):
        self.producer.flush()

    def poll(self):
        self.producer.poll(0)

    def produce(self):
        self.producer.produce(topic=self.topic_name,
                              value=json.dumps(self.json_message).encode('utf-8'),
                              callback=self.callback_report)