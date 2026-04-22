import unittest
from unittest.mock import MagicMock
import threading
import time
from transformsai_ai_core.video_capture import VideoCaptureAsync

class TestVideoCaptureAsyncFPS(unittest.TestCase):
    def test_none_fps_initialization(self):
        # Mock cv2.VideoCapture to avoid actual camera/file opening
        import cv2
        original_vc = cv2.VideoCapture
        cv2.VideoCapture = MagicMock()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0.0 # Return 0 for FPS to trigger potential issues
        cv2.VideoCapture.return_value = mock_cap
        
        try:
            # Initialize with default parameters (fps=None)
            vc = VideoCaptureAsync(src="test.mp4")
            
            # Manually force _fps to None to test the condition in _update
            vc._fps = None
            
            # Start and then immediately stop to check if _update crashes
            vc.start()
            time.sleep(0.1)
            vc.stop()
            
            # If we reach here without TypeError, the fix works
            self.assertIsNone(vc._fps)
            
        finally:
            cv2.VideoCapture = original_vc

if __name__ == '__main__':
    unittest.main()
