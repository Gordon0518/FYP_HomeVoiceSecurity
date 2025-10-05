import torch
import soundfile as sf
from transformers import pipeline
import wave
import io
import numpy as np

# Global device and model (loaded once for efficiency)
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_pipe = pipeline("automatic-speech-recognition",
                    model="openai/whisper-small",
                    device=device)


def speech_to_text(audio_data):
    """
    Convert raw audio data to text using Whisper model.
    Expects audio_data as bytes (WAV format).
    Returns transcribed text or error message.
    """
    if audio_data is None:
        return "No speech detected"

    try:
        # Convert to WAV format
        with io.BytesIO() as temp_file:
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_data)
            wf.close()
            temp_file.seek(0)
            audio_array, sample_rate = sf.read(temp_file, dtype='float32')

        # Debug print for audio info
        print(f"Audio array shape: {audio_array.shape}, Sample rate: {sample_rate}")
        print(f"Audio min: {np.min(audio_array):.3f}, max: {np.max(audio_array):.3f}")

        # Check audio volume (if too quiet, warn)
        max_amp = np.max(np.abs(audio_array))
        if max_amp < 0.01:
            print("Warning: Audio volume is low. Check microphone sensitivity.")

        # Whisper transcription with correct language parameter
        result = stt_pipe(audio_array, generate_kwargs={"language": "en"})
        text = result["text"] if result["text"] else "No text recognized"

        # Debug print for result
        print(f"Transcribed text: {text}")
        return text

    except Exception as e:
        print(f"STT Error: {str(e)}")
        return f"STT Error: {str(e)}"