import cv2
import threading
import paho.mqtt.client as mqtt
import time
import base64
import numpy as np

# --- Information About Script ---
__name__ = "Legacy MQTT Streamer"
__version__ = "1.0.5" 
__author__ = "TransformsAI"

class StreamPublisher:

    def __init__(self, topic, target_width=640, start_stream=True, host="127.0.0.1", port=1883, jpeg_quality=80) -> None:
        """
        Construct a new 'stream_publisher' object to broadcast a video stream using Mosquitto_MQTT

        :param topic: MQTT topic to send Stream
        :param target_width: The desired width for the resized image. Height will be calculated to maintain aspect ratio. Default is 640.
        :param start_stream: start streaming while making object, default True, else call object.start_streaming()
        :param host: IP address of Mosquitto MQTT Broker
        :param Port: Port at which Mosquitto MQTT Broker is listening

        :return: returns nothing
        """

        self.client = mqtt.Client()  # create new instance

        print(f"Connecting to MQTT broker at {host}:{port}")
        try:
            self.client.connect(host, port)
            print("MQTT connection successful.")
        except Exception as e:
            print(f"MQTT connection failed: {e}")


        self.topic = topic
        self.img = None
        self.stop_stream = False
        self.start_stream = start_stream
        self.streaming_thread = None
        self.read_lock = threading.Lock()
        self.target_width = target_width # Store the target width
        self.jpeg_quality = jpeg_quality

        if self.start_stream:
            self.start_streaming()

    def start_streaming(self):
        if self.streaming_thread is not None and self.streaming_thread.is_alive():
            print("Streaming thread is already running.")
            return

        self.start_stream = True
        self.stop_stream = False
        self.streaming_thread = threading.Thread(target=self.stream)
        self.streaming_thread.daemon = True # Allow program to exit even if thread is running
        self.streaming_thread.start()
        print("Attempting to start streaming thread.")


    def stop_streaming(self):
        if self.streaming_thread is None or not self.streaming_thread.is_alive():
            print("Streaming thread is not running.")
            return

        print("Stopping streaming thread...")
        self.stop_stream = True
        self.start_stream = False
        # Give the thread a moment to finish its loop iteration
        time.sleep(0.1)
        if self.streaming_thread.is_alive():
             self.streaming_thread.join(timeout=1.0) # Wait for thread to finish, with a timeout
             if self.streaming_thread.is_alive():
                 print("Warning: Streaming thread did not stop gracefully.")
             else:
                 print("Streaming thread stopped.")
        else:
             print("Streaming thread was already stopped.")


    def updateFrame(self, frame):
        """
        Updates the frame to be streamed. This method should be called
        by the source providing the frames (e.g., a video capture loop).
        """
        if frame is not None and frame.size > 0:
            with self.read_lock:
                # Ensure the frame is a valid image before copying
                if isinstance(frame, np.ndarray) and frame.ndim >= 2:
                     self.img = frame.copy()
                else:
                     print("Warning: updateFrame received invalid frame data.")
        # else:
        #     print("Warning: updateFrame received None or empty frame.") # Optional: uncomment for debugging


    def liveStream(self, status):
        """
        Controls whether the stream thread should actively process and publish frames.
        The thread itself keeps running, but it pauses processing if status is False.
        """
        self.start_stream = status
        if status:
            print("Live streaming enabled.")
        else:
            print("Live streaming paused.")


    def stream(self):
        """
        The main streaming loop running in a separate thread.
        Reads the latest frame, resizes it maintaining aspect ratio,
        encodes it, and publishes it via MQTT.
        """
        print("Streaming thread started.")
        while not self.stop_stream:
            frame_to_process = None # Use a temporary variable

            # --- Acquire and copy frame ---
            # Only acquire lock and copy if we are actively streaming AND there's a frame
            if self.start_stream:
                 with self.read_lock:
                    if self.img is not None:
                        # Make a copy inside the lock
                        frame_to_process = self.img.copy()
                        self.img = None # Clear the shared image immediately after copy

            # --- Process and Publish (outside the lock) ---
            if frame_to_process is not None:
                try:
                    # Get original dimensions
                    original_height, original_width = frame_to_process.shape[:2]

                    # Check if dimensions are valid for resizing
                    if original_width > 0 and original_height > 0:
                        # Calculate target height maintaining aspect ratio
                        # target_height = (target_width * original_height) / original_width
                        target_height = int((self.target_width * original_height) / original_width)

                        # Ensure height is at least 1 pixel to avoid errors
                        if target_height == 0:
                            target_height = 1

                        new_size = (self.target_width, target_height)

                        # Resize the image
                        img_resized = cv2.resize(frame_to_process, new_size)

                        # Encode the resized image
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        result, img_str = cv2.imencode('.jpg', img_resized, encode_param)

                        if result:
                            # Convert to base64 and publish
                            image_base64 = base64.b64encode(img_str).decode('utf-8')
                            # Use client.publish with qos=0 for speed, or qos=1/2 for reliability
                            self.client.publish(self.topic, image_base64, qos=0)
                            # print(f"Published frame to {self.topic}") # Optional: uncomment for debugging
                        else:
                            print("Warning: Failed to encode frame.")

                    else:
                        print(f"Warning: Received frame with invalid dimensions ({original_width}x{original_height}), skipping resize/publish.")

                except cv2.error as e:
                    print(f"OpenCV error during processing: {e}")
                except Exception as e:
                    print(f"Error during frame processing/publishing: {e}")

            elif self.stop_stream:
                break 

            time.sleep(0.03)

        print("Streaming thread stopped.")
        # Optional: Disconnect MQTT client when thread stops
        self.client.disconnect()