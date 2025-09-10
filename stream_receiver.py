import cv2
import threading
import numpy as np
import paho.mqtt.client as mqtt
import json

# --- Information About Script ---
__name__ = "Legacy MQTT Message Reciever"
__version__ = "0.0.6" 
__author__ = "TransformsAI"

class StreamReceiver:

    def __init__(self, callback_update_settings, sn='',host="127.0.0.1",port=1883):

        """
        Construct a new 'stream_receiver' object to retreive a video stream using Mosquitto_MQTT

        :param topic: MQTT topic to send Stream         
        :param host:  IP address of Mosquitto MQTT Broker
        :param Port:  Port at which Mosquitto MQTT Broker is listening
        
        :return: returns nothing

        : use " object.frame  "  it contains latest frame received
        """
        self.callback_update_settings = callback_update_settings
        self.sn = sn
        self.topic="settings_"+self.sn
        self.frame=None  # empty variable to store latest message received
        
        self.client = mqtt.Client()  # Create instance of client 

        self.client.on_connect = self.on_connect  # Define callback function for successful connection
        self.client.message_callback_add(self.topic,self.on_message)
        
        self.client.connect(host,port)  # connecting to the broking server
        
        t=threading.Thread(target=self.subscribe)       # make a thread to loop for subscribing
        t.start() # run this thread
        
    def subscribe(self):
        self.client.loop_forever() # Start networking daemon
        
    def on_connect(self,client, userdata, flags, rc):  # The callback for when the client connects to the broker
        client.subscribe(self.topic)  # Subscribe to the topic, receive any messages published on it
        print("Subscribing to topic :",self.topic)

    def on_message(self,client, userdata, msg):  # The callback for when a PUBLISH message is received from the server.

            # Decode payload bytes into a string
            json_string = msg.payload.decode('utf-8')

            # Parse JSON string into a Python dictionary
            config_requested = json.loads(json_string)

            
            self.config = {}
            with open("config_"+self.sn+".json", 'r') as f:
                self.config = json.load(f)

            if "settings_requested" in config_requested:
                 response_json = json.dumps(self.config)
                 self.client.publish("settings_" + self.sn+"_response", payload=str(response_json))
            else:
                # Iterate through keys in config_requested   
                for key in config_requested:
                    # Check if the key exists in self.config
                    if key in self.config and key != 'local_ip':
                        # Update the value in self.config with the value from config_requested
                        self.config[key] = config_requested[key]

                with open("config_" + self.sn+".json", 'w') as f:
                    json.dump(self.config, f, indent=4)
                    response_json = json.dumps(self.config)
                    self.client.publish("settings_" + self.sn+"_response", payload=response_json)
                self.callback_update_settings("config_" + self.sn+".json")