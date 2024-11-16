# Add the parent directory of the packages to the Python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'face-emotion-detection')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'audio-analysis')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tone-analysis')))


from audio_analysis.audio_to_chunks import load_chunks_model, diarize_audio, get_intervals, split_audio
from audio_analysis.audio_to_text import load_text_model, transcribe_directory
from tone_analysis.tone_analysis import predict_emotion
from face_emotion_detection.detect import detect_and_predict_emotions

# Setup directories
audio_file = 'path_to_audio_file.wav'
video_file = 'path_to_video_file.mp4'
output_dir = 'output_chunks'
transcription_file = 'Chunks_transcriptions.json'
emotion_results_file = 'emotion_results.json'

# # Load models
# chunks_model = load_chunks_model()
# text_model = load_text_model()

