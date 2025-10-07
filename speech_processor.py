import torch
import soundfile as sf
from transformers import pipeline
import wave

import io
import numpy as np
import librosa  # For resampling

# Global device and model (loaded once for efficiency)
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_pipe = pipeline("automatic-speech-recognition",
                    model="openai/whisper-small",
                    device=device)


def speech_to_text(audio_data, original_rate=48000):
    """
    Convert raw audio data to text using Whisper model.
    Expects audio_data as bytes (WAV format) at original_rate (e.g., 48000 Hz).
    Resamples to 16000 Hz for Whisper.
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
            wf.setframerate(original_rate)  # Use original rate
            wf.writeframes(audio_data)
            wf.close()
            temp_file.seek(0)
            audio_array, sample_rate = sf.read(temp_file, dtype='float32')

        # Debug print for audio info
        print(f"Audio array shape: {audio_array.shape}, Original Sample Rate: {sample_rate} Hz")
        print(f"Audio min: {np.min(audio_array):.3f}, max: {np.max(audio_array):.3f}")

        # Check audio volume (if too quiet, warn)
        max_amp = np.max(np.abs(audio_array))
        if max_amp < 0.01:
            print("Warning: Audio volume is low. Check microphone sensitivity.")

        # Resample to 16000 Hz for Whisper if necessary
        if sample_rate != 16000:
            print(f"Resampling from {sample_rate} Hz to 16000 Hz...")
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Whisper transcription with correct language parameter
        result = stt_pipe(audio_array, generate_kwargs={"language": "en"})
        text = result["text"] if result["text"] else "No text recognized"

        # Debug print for result
        print(f"Transcribed text: {text}")
        return text

    except Exception as e:
        error_msg = "STT Error: " + str(e)
        print(error_msg)
        return error_msg