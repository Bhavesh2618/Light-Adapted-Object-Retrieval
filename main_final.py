import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import speech_recognition as sr
import io
import cv2
import torch
from ultralytics import YOLO
import threading

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
def load_model(model_path):
    model = YOLO(model_path)
    model.to(device)
    return model

# Function to perform object detection on a frame
def detect_objects(model, frame, target_object):
    # Resize frame to a smaller resolution for faster processing
    frame_resized = cv2.resize(frame, (480, 480))  # Reduced resolution
    results = model.predict(source=frame_resized, imgsz=640, conf=0.47, device=device, verbose=False)  # Confidence threshold
    
    detected_objects = []
    output_frame = frame.copy()  # Create a copy of the original frame for drawing

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)].lower()  # Convert label to lowercase
            
            if label == target_object:
                detected_objects.append(label)
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                
                # Scale the bounding box coordinates back to the original frame size
                scale_x = frame.shape[1] / 480  # Original width / resized width
                scale_y = frame.shape[0] / 480  # Original height / resized height
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                
                # Draw bounding box and label on the original frame
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                cv2.putText(output_frame, target_object, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return output_frame, detected_objects

# Record audio in a non-blocking manner
def record_audio(duration, samplerate=16000):
    print("Listening for command...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio_data, samplerate

# Recognize speech in a separate thread
def recognize_speech_thread(valid_commands, current_command):
    duration = 4  # Duration for recording audio
    samplerate = 16000  # Sample rate for audio recording

    recognizer = sr.Recognizer()

    while True:
        # Record audio
        audio_data, samplerate = record_audio(duration, samplerate)
        
        # Recognize speech
        audio_buffer = io.BytesIO()
        wavfile.write(audio_buffer, samplerate, audio_data)
        audio_buffer.seek(0)
        
        try:
            with sr.AudioFile(audio_buffer) as source:
                audio = recognizer.record(source)
                command = recognizer.recognize_google(audio).lower()
                if command in valid_commands:
                    print(f"Recognized command: {command}")
                    current_command[0] = command  # Update the current command
                elif command == "exit":
                    print("Exiting program...")
                    current_command[0] = "exit"
                    break
                else:
                    print("Command not recognized. Please try again.")
        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
        except sr.RequestError as e:
            print(f"Error connecting to Google API: {e}")

# Play video and detect objects
def play_video_with_detection(video_path, model, valid_commands):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Create a named window for the video
    cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)  # Ensure the window is resizable
    cv2.resizeWindow("Video Detection", 800, 600)  # Set initial window size

    # Initialize variables
    current_command = [""]  # Shared variable to store the current command
    previous_command = ""  # Variable to track the previously detected object
    detected_log = set()  # Track which objects have already been logged

    # Start a thread for continuous speech recognition
    threading.Thread(target=lambda: recognize_speech_thread(valid_commands, current_command)).start()

    while True:  # Infinite loop for continuous video playback
        ret, frame = cap.read()
        if not ret:
            # Reset the video capture to the beginning if it reaches the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                print("Error resetting video playback.")
                break

        # Perform object detection for the current command
        target_object = current_command[0]
        if target_object == "exit":
            print("Stopping video playback...")
            break
        elif target_object:
            detected_frame, detected_objects = detect_objects(model, frame, target_object)
            
            # Log the detected object only once
            if detected_objects and target_object not in detected_log:
                print(f"Detected object: {target_object}")
                detected_log.add(target_object)  # Mark this object as logged
            
            previous_command = target_object  # Update the previous command
        elif previous_command:
            # If no new command is recognized, keep displaying the previous command's detection
            detected_frame, detected_objects = detect_objects(model, frame, previous_command)
            
            # Log the detected object only once
            if detected_objects and previous_command not in detected_log:
                print(f"Detected object: {previous_command}")
                detected_log.add(previous_command)  # Mark this object as logged
        else:
            detected_frame = frame  # No detection if no command is active

        # Display the frame
        cv2.imshow("Video Detection", detected_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    valid_commands = ["forceps", "hemostat", "mayo", "scalpel", "syringe", "stitch scissors", "episiotomy scissors", "cotton", "gloves", "exit"]

    # Paths
    video_path = r"C:\Users\HP\Downloads\data videos\v 5.mp4"  # Change to your video path
    model_path = r"C:\Users\HP\Downloads\Trained Results Kaggle\YOLO11n_110epochs\weights\best.pt" # Change to your best.pt path

    # Load the YOLO model
    model = load_model(model_path)

    try:
        print("Playing video. Listening for voice commands...")
        play_video_with_detection(video_path, model, valid_commands)
    except KeyboardInterrupt:
        print("\nExiting program...")
