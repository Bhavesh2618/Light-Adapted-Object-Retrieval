import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import speech_recognition as sr
import io
import threading

def record_audio(duration, samplerate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    return audio_data, samplerate

def recognize_speech(audio_data, samplerate, valid_commands):
    recognizer = sr.Recognizer()
    audio_buffer = io.BytesIO()
    wavfile.write(audio_buffer, samplerate, audio_data)
    audio_buffer.seek(0)
    
    with sr.AudioFile(audio_buffer) as source:
        audio = recognizer.record(source)

    # Define a function to run the speech recognition in a thread
    def recognize():
        nonlocal command
        try:
            command = recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            command = ""
        except sr.RequestError as e:
            print(f"Error connecting to Google API: {e}")
            command = ""

    # Initialize the command variable
    command = None

    # Create and start a thread for speech recognition
    recognition_thread = threading.Thread(target=recognize)
    recognition_thread.start()

    # Wait for the thread to finish or timeout after 5 seconds
    recognition_thread.join(timeout=5)

    # If the thread is still alive after the timeout, it means recognition took too long
    if recognition_thread.is_alive():
        print("Speech recognition timed out. Please try again.")
        return ""

    # Check if the recognized command is one of the valid commands
    if command in valid_commands:
        print(f"Recognized command: {command}")
        return command
    else:
        print("Command not recognized. Please try again.")
        return ""

def mock_send_command(command):
    print(f"Mock sending command: {command}")

if __name__ == "__main__":
    duration = 4  # Record for 4 seconds
    valid_commands = ["forceps", "hemostat", "mayo", "scalpel", "syringe", "stitch", "episiotomy", "cotton", "gloves", "exit"]

    try:
        while True:
            # Record audio
            audio_data, samplerate = record_audio(duration)
            
            # Check if audio data is empty
            if audio_data.size == 0:
                print("No audio data recorded. Please check your microphone.")
                continue
            
            # Recognize speech
            user_input = recognize_speech(audio_data, samplerate, valid_commands)
            
            # Process only if the command is valid
            if user_input:
                mock_send_command(user_input)
            
            # Break the loop if the recognized command is 'exit'
            if user_input == "exit":
                break
    except KeyboardInterrupt:
        print("\nExiting program...")
