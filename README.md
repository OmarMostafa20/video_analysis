# Video Analysis

## Project Overview
This project integrates three main components to analyze meeting video and audio data to detect and predict emotions from speech and facial expressions. The components include:

1. **Audio Analysis**: Diarization, chunking, and transcription of audio.
2. **Tone Analysis**: Emotion detection from audio chunks.
3. **Face Emotion Detection**: Face detection and emotion prediction from video frames.

---

## Architecture and Flow

### Sequential Flow
1. **Audio Processing**:
   - **Diarization and Chunking**: Segment audio based on speaker diarization.
   - **Transcription**: Convert audio chunks into text.
2. **Tone Analysis**:
   - Predict emotions in transcribed audio chunks.
3. **Face Emotion Detection**:
   - Detect faces and predict emotions from video frames.

### Parallel Flow
- **Audio Processing** and **Face Emotion Detection** can run in parallel as they are independent tasks.
- **Tone Analysis** is performed after Audio Processing since it depends on the transcribed text.

---

## Models Used
- **Speaker Diarization**: `pyannote.audio` pre-trained speaker diarization model.
- **Speech-to-Text**: OpenAI's Whisper large model.
- **Audio Emotion Detection**: Custom ResNet50v2 model trained on combined English and Arabic datasets.
- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks).
- **Facial Emotion Detection**: Custom CNN model (`EmotionDetector.h5`).

---

## Datasets Used
### Speech Emotion Recognition
- **English**: RAVDESS, CREMA-D, TESS, SAVEE.
- **Arabic**: EYASE dataset.

### Face Emotion Recognition
- FER-2013 dataset for training the CNN model.

---

## Metrics for Measurement
1. **Accuracy**: Proportion of correctly predicted emotions.
2. **Precision**: Correct positive predictions divided by total positive predictions.
3. **Recall**: Correct positive predictions divided by total actual positives.
4. **F1 Score**: Harmonic mean of precision and recall.
5. **Latency**: Time taken to process each component.

---

## Performance Evaluation
1. **Latency**: Measure the time taken for each step in the process.
2. **Resource Utilization**: Monitor CPU, GPU, and memory usage during processing.
3. **Accuracy**: Evaluate the accuracy of emotion detection models.
