import unittest
from tone_analysis.tone_analysis import predict_emotion, preprocess_sample
import numpy as np

class TestToneAnalysis(unittest.TestCase):

    def test_preprocess_sample(self):
        sample = np.random.rand(128)
        processed_sample = preprocess_sample(sample)
        self.assertEqual(processed_sample.shape, (1, 128, 1))

    def test_predict_emotion(self):
        sample = np.random.rand(128)
        emotion = predict_emotion(sample)
        self.assertIn(emotion, ['angry', 'happy', 'neutral', 'sad'])

if __name__ == '__main__':
    unittest.main()
