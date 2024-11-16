import whisper
from pydub import AudioSegment
import numpy as np
import json
import os

def load_text_model():
    # Load Whisper model
    model = whisper.load_model("large")
    return model

def load_audio(file_path):
    # Load the audio file and convert it to the required format
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return audio_samples, len(audio_samples)

def transcribe_audio(model, file_path):
    audio_samples, _ = load_audio(file_path)
    result = model.transcribe(audio_samples)
    return result

def transcribe_directory(model, directory_path='output_chunks', output_file='Chunks_transcriptions.json'):
    # List to store transcription results
    transcriptions = []

    # Process each audio file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            transcription_result = transcribe_audio(model, file_path)
            transcriptions.append({
                "Chunk_Name": filename,
                "Transcript_Text": transcription_result['text']
            })

    # Save the results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, indent=4, ensure_ascii=False)

    print(f"Transcriptions saved to '{output_file}'")

def main():
    model = load_text_model()
    transcribe_directory(model)

if __name__ == "__main__":
    main()
