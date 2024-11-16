import unittest
from face_emotion_detection.detect import detect_and_predict_emotions

class TestFaceEmotionDetection(unittest.TestCase):

    def test_detect_and_predict_emotions(self):
        video_path = 'path_to_test_video_file.wav'
        try:
            detect_and_predict_emotions(video_path)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Test failed due to: {e}")

if __name__ == '__main__':
    unittest.main()
