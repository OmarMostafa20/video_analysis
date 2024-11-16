import numpy as np
from keras import models
from mtcnn import MTCNN
import cv2

# Load the model
model = models.load_model('face_emotion_detection/EmotionDetector.h5')

# Define the class labels
class_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Function to detect faces and predict emotions
def detect_and_predict_emotions(video_path):

    # Open a connection to the video
    video_capture = cv2.VideoCapture(video_path)
    detector = MTCNN()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces
        results = detector.detect_faces(frame)

        if results:
            for result in results:
                bounding_box = result['box']
                x, y, width, height = bounding_box
                face = frame[y:y+height, x:x+width]

                # Preprocess the face
                face = cv2.resize(face, (48, 48))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = face / 255.0
                face = np.reshape(face, (1, 48, 48, 1))

                # Predict emotion
                emotion_prediction = model.predict(face)
                max_index = np.argmax(emotion_prediction[0])
                emotion = class_labels[max_index]

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: detect-emotions <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    detect_and_predict_emotions(video_path)

if __name__ == "__main__":
    main()
