# tone_analysis.py

# Import Libraries
from keras import models
import pickle
import numpy as np
import json
import os
from tone_analysis.helper import *

#######################################################################################################################
#######################################################################################################################

# Load Models
model = models.load_model('en_ar_ser_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

#######################################################################################################################
#######################################################################################################################

def preprocess_sample(sample):
    sample = scaler.transform([sample])
    sample = np.expand_dims(sample, axis=2)
    return sample

def predict_emotion(sample):
    sample = preprocess_sample(sample)
    prediction = model.predict(sample)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def main():
    # Example usage:
    # Directory containing audio files
    audio_dir = "output_chunks"

    # List to store results
    results = []

    # Expected input shape for the model
    expected_input_shape = model.input_shape[1]

    # Loop through each file in the directory and predict emotions
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(audio_dir, file_name)

            features = extract_features_from_audio(file_path)

            # Check if the extracted features match the expected input shape
            if features.shape[0] != expected_input_shape:
                # Handle mismatch by padding or truncating
                if features.shape[0] < expected_input_shape:
                    # Pad with zeros
                    features = np.pad(features, (0, expected_input_shape - features.shape[0]), 'constant')
                else:
                    # Truncate the features
                    features = features[:expected_input_shape]

            predicted_emotion = predict_emotion(features)
            
            if predicted_emotion in ['fear', 'disgust']:
                predicted_emotion = 'angry'

            if predicted_emotion in ['surprise']:
                predicted_emotion = 'happy'

            # if predicted_emotion in ['fear', 'disgust','angry','sad']:
            #     predicted_emotion = 'negative'

            # if predicted_emotion in ['happy', 'surprise']:
            #     predicted_emotion = 'postive'

            results.append({"audio_file": file_name, "predicted_emotion": predicted_emotion})

    # Save the results to a JSON file
    with open("example-emotion.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Emotion predictions saved to example-emotion-predictions.json")

if __name__ == "__main__":
    main()
