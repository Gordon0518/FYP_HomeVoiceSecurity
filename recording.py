import pyaudio
import webrtcvad
import collections

# Recording parameters (shared across the project)
CHUNK = 480  # Required for webrtcvad
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
MICROPHONE_INDEX = 0  # Change this if microphone is not default
VAD_MODE = 3  # Aggressive mode for noise filtering


def record_vad_audio():
    """
    Record 5-second audio with VAD (Voice Activity Detection) to filter noise.
    Returns raw audio data as bytes.
    """
    vad = webrtcvad.Vad(VAD_MODE)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                    input_device_index=MICROPHONE_INDEX, frames_per_buffer=CHUNK)

    frames = []
    ring_buffer = collections.deque(maxlen=100)  # Buffer for 1 second of audio
    triggered = False
    voiced_frames = []
    silence_duration = 0.5  # Stop after 0.5s silence

    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        ring_buffer.append(data)
        is_speech = vad.is_speech(data, sample_rate=RATE)

        if not triggered:
            if is_speech:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(data)
            if not is_speech:
                silence_duration -= CHUNK / RATE
                if silence_duration <= 0:
                    break

    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(voiced_frames) if voiced_frames else None