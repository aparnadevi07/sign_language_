import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from langid import classify  # Auto-detect language

# Path to Vosk model
VOSK_MODEL_PATH = "vosk-model"

# Check if the model exists
if not os.path.exists(VOSK_MODEL_PATH):
    raise Exception("❌ Vosk model not found! Download from https://alphacephei.com/vosk/models and extract it.")

# Load Vosk model
model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)  # 16kHz sample rate

# Initialize Microphone
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
stream.start_stream()

print("🎤 Speak something...")

while True:
    data = stream.read(4096)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        text = result.get("text", "")

        # Auto-detect language
        detected_lang, _ = classify(text)
        lang_map = {"ta": "Tamil", "hi": "Hindi", "te": "Telugu", "en": "English"}
        lang_name = lang_map.get(detected_lang, "Unknown")

        print(f"📝 {text}  [{lang_name}]")
