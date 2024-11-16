import cv2
from mtcnn import MTCNN

def process_video(video_path, confidence_threshold=0.95):
    # Initialize MTCNN detector
    detector = MTCNN()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
        detections = detector.detect_faces(rgb_frame)
        
        for detection in detections:
            if detection['confidence'] >= confidence_threshold:
                x, y, width, height = detection['box']
                # Draw bounding box around detected face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Video', frame)
        
        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()

# Example usage
input_video_path = 'sample.mp4'
process_video(input_video_path)
