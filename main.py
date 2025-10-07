import pyaudio

def list_microphones():
    """List available microphones with PyAudio indices."""
    p = pyaudio.PyAudio()
    print("Available PyAudio Microphones:")
    print("-" * 50)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:  # Input device (mic)
            print(f"PyAudio Index {i}: {info['name']}")
            print(f"  - Max Channels: {info['maxInputChannels']}")
            print(f"  - Default Sample Rate: {info['defaultSampleRate']} Hz")
            print("-" * 30)
    p.terminate()

if __name__ == "__main__":
    list_microphones()