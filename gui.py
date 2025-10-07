import tkinter as tk
from tkinter import ttk, messagebox
import threading
from recording import record_vad_audio  # Import recording function
from speech_processor import speech_to_text  # Import STT function


class SimpleGUIRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Voice Recorder for Jetson Nano")
        self.root.geometry("400x300")

        # GUI elements
        self.label = tk.Label(root, text="Press 'Start Recording' to speak for 5 seconds", font=("Arial", 12))
        self.label.pack(pady=20)

        self.button = tk.Button(root, text="Start Recording", command=self.start_recording,
                                bg="green", fg="white", font=("Arial", 14))
        self.button.pack(pady=10)

        self.result_label = tk.Label(root, text="Result will appear here", font=("Arial", 10), wraplength=350)
        self.result_label.pack(pady=20)

        self.status_label = tk.Label(root, text="Status: Ready", font=("Arial", 10))
        self.status_label.pack()

    def start_recording(self):
        self.button.config(state="disabled")
        self.status_label.config(text="Status: Recording...")
        self.root.update()

        # Record in a separate thread to avoid freezing GUI
        thread = threading.Thread(target=self.record_and_process)
        thread.start()

    def record_and_process(self):
        try:
            # Record audio with VAD

            audio_data = record_vad_audio()

            # Process with STT
            text = speech_to_text(audio_data)

            # Print to console
            print("Recognized text:", text)

            # Update GUI
            self.root.after(0, self.update_gui, text)

        except Exception as e:
            print(f"Error: {str(e)}")
            self.root.after(0, self.update_gui, f"Error: {str(e)}")

    def update_gui(self, text):
        self.result_label.config(text=f"Recognized text: {text}")
        self.status_label.config(text="Status: Ready")
        self.button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleGUIRecorder(root)
    root.mainloop()