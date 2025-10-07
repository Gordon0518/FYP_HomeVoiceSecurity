import pyaudio
import webrtcvad
import collections
import numpy as np

# Recording parameters (shared across the project)
CHUNK = 960  # Adjusted for 48000 Hz (20ms frame for VAD)
FORMAT = pyaudio.paInt16
RATE = 48000  # Use headset default rate for better compatibility
RECORD_SECONDS = 5
VAD_MODE = 2  # Less aggressive mode for better detection with low-volume mics
MICROPHONE_INDEX = 21  # Change this to your G435 headset PyAudio index (e.g., from list_microphones)


def get_available_microphones():
    """
    List available microphones and their channel support.
    Returns a list of dicts with device info.
    """
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # Input device
            devices.append({
                'index': i,
                'name': info['name'],
                'max_channels': info['maxInputChannels'],
                'default_sample_rate': info['defaultSampleRate']
            })
    p.terminate()
    return devices


def record_vad_audio():
    """
    Record 5-second audio with VAD (Voice Activity Detection) to filter noise.
    Uses global MICROPHONE_INDEX for device selection.
    Returns raw audio data as bytes at original rate.
    """
    mic_index = MICROPHONE_INDEX  # Use global index (easy to modify at top)
    p = pyaudio.PyAudio()
    device_info = p.get_device_info_by_index(mic_index)
    supported_channels = int(device_info['maxInputChannels'])

    # For headsets, often stereo (2 channels), use what the device supports
    channels = supported_channels  # Use device's max channels
    if channels == 0:
        raise ValueError(f"No input channels on device {mic_index}. Check 'arecord -l'.")

    print(f"Using microphone: {device_info['name']} (Index {mic_index}), Channels: {channels}, Sample Rate: {RATE} Hz")

    stream = p.open(format=FORMAT, channels=channels, rate=RATE, input=True,
                    input_device_index=mic_index, frames_per_buffer=CHUNK)

    vad = webrtcvad.Vad(VAD_MODE)
    frames = []
    ring_buffer = collections.deque(maxlen=200)  # Adjusted for 48000 Hz (1 second buffer)
    triggered = False
    voiced_frames = []
    silence_duration = 0.5  # Stop after 0.5s silence
    speech_frames_count = 0  # Debug: count speech frames

    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        ring_buffer.append(data)

        # For VAD, always convert to mono (headsets are often stereo)
        audio_mono = np.frombuffer(data, dtype=np.int16)
        if channels > 1:
            audio_mono = audio_mono.reshape(-1, channels).mean(axis=1)
        data_mono = audio_mono.astype(np.int16).tobytes()
        is_speech = vad.is_speech(data_mono, sample_rate=RATE)

        if is_speech:
            speech_frames_count += 1  # Count speech frames for debug

        if not triggered:
            if is_speech:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
                print(f"Speech detected at frame {i}, starting recording...")
        else:
            voiced_frames.append(data)
            if not is_speech:
                silence_duration -= CHUNK / RATE
                if silence_duration <= 0:
                    print(f"Silence detected, stopping after {i} frames. Speech frames: {speech_frames_count}")
                    break

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not voiced_frames:
        print(
            f"No speech detected. Total frames: {int(RATE / CHUNK * RECORD_SECONDS)}, Speech frames: {speech_frames_count}")
        return None

    print(f"Recorded {len(voiced_frames)} voiced frames at {RATE} Hz. Total speech frames: {speech_frames_count}")
    return b''.join(voiced_frames)