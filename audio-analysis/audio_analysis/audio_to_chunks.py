from pyannote.audio import Pipeline
import torchaudio
import re
import os
import torch
from pydub import AudioSegment

def load_chunks_model():
    # Load Model
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"
    )

    # Send pipeline to GPU (when available)
    pipeline.to(torch.device("cuda"))
    return pipeline

def diarize_audio(pipeline, audio_file):
    # Pre-loading audio files in memory may result in faster processing
    waveform, sample_rate = torchaudio.load(audio_file)
    diarization_result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    return diarization_result

def get_intervals(diarization_result):
    diarization_string = f"{diarization_result}"

    # Pattern to extract time stamps and speaker
    pattern = r"\[\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\]\s*[A-Z]+\s*(SPEAKER_\d+)"

    # Find all matches
    matches = re.findall(pattern, diarization_string)

    # To store intervals for each speaker change
    intervals = []
    current_speaker = None
    start_time = None

    for start, end, speaker in matches:
        if speaker != current_speaker:
            if current_speaker is not None:
                intervals.append({"Current Speaker": current_speaker, "Start time": start_time, "End time": prev_end_time})

            start_time = start
            current_speaker = speaker
        prev_end_time = end

    # Append the last segment
    if current_speaker is not None:
        intervals.append({"Current Speaker": current_speaker, "Start time": start_time, "End time": prev_end_time})

    return intervals

def split_audio(audio_file, intervals, output_dir='output_chunks'):
    audio = AudioSegment.from_file(audio_file)

    def time_to_millis(t):
        """Convert time string to milliseconds."""
        h, m, s = t.split(':')
        s, ms = s.split('.')
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

    # Create a directory for output chunks if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the audio based on intervals
    for index, interval in enumerate(intervals):
        start_time = time_to_millis(interval['Start time'])
        end_time = time_to_millis(interval['End time'])
        chunk = audio[start_time:end_time]
        # Export audio chunk
        chunk.export(f'{output_dir}/{interval["Current Speaker"]}_chunk_{index+1}.wav', format='wav')

    print("Audio chunks have been created successfully.")

def main():
    pipeline = load_chunks_model()
    audio_file = 'path_to_your_audio_file.wav'
    diarization_result = diarize_audio(pipeline, audio_file)
    intervals = get_intervals(diarization_result)
    split_audio(audio_file, intervals)

if __name__ == "__main__":
    main()
