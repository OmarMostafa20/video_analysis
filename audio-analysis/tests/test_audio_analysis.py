import unittest
import os
from audio_analysis.audio_to_chunks import load_chunks_model, diarize_audio, get_intervals, split_audio
from audio_analysis.audio_to_text import load_text_model, transcribe_directory

class TestAudioAnalysis(unittest.TestCase):

    def setUp(self):
        self.audio_file = 'path_to_your_audio_file.wav'
        self.output_dir = 'test_output_chunks'
        self.output_file = 'test_Chunks_transcriptions.json'
        self.chunks_model = load_chunks_model()
        self.text_model = load_text_model()

    def tearDown(self):
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_audio_analysis_process(self):
        # Diarize audio
        diarization_result = diarize_audio(self.chunks_model, self.audio_file)
        intervals = get_intervals(diarization_result)

        # Split audio into chunks
        split_audio(self.audio_file, intervals, self.output_dir)

        # Check if chunks are created
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertGreater(len(os.listdir(self.output_dir)), 0)

        # Transcribe audio chunks
        transcribe_directory(self.text_model, directory_path=self.output_dir, output_file=self.output_file)

        # Check if transcription file is created
        self.assertTrue(os.path.exists(self.output_file))

if __name__ == '__main__':
    unittest.main()
