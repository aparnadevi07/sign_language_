import os
import cv2
import speech_recognition as sr
import moviepy.editor as mp

# Define the exact video path
video_path = r"C:\signlang\video_sign.mp4"

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit()

print("Video file found. Processing...")

# Load the video
video = mp.VideoFileClip(video_path)

# Extract audio from the video
audio_path = "extracted_audio.wav"
video.audio.write_audiofile(audio_path)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Convert audio to text
with sr.AudioFile(audio_path) as source:
    print("Extracting text from audio...")
    audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print("Extracted Text:\n", text)
        
        # Save the text to a file
        with open("recognized_text.txt", "w") as file:
            file.write(text)
        print("Text saved to recognized_text.txt")
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError:
        print("API request error")

print("Processing complete.")

#adtt