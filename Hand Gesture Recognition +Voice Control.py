import cv2
import mediapipe as mp
import numpy as np
from joblib import dump, load
import tkinter as tk
import pyautogui
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pycaw.pycaw import AudioUtilities , IAudioEndpointVolume
from ctypes import cast , POINTER
from comtypes import CLSCTX_ALL
import threading
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from matplotlib.backends.backend_pdf import PdfPages  # Needed for PDF saving
import speech_recognition as sr
import pythoncom
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
drawing_utils = mp.solutions.drawing_utils

def extract_features(landmarks, w, h):
    base = landmarks[0]
    features = []
    for lm in landmarks:
        x = (lm.x - base.x) * w
        y = (lm.y - base.y) * h
        features.extend([x, y])
    return features

class GestureApp:
    def __init__(self, window):
        self.labels = None
        self.y_pred = None
        self.y_test = None
        self.banner_image = None
        self.thread = None
        self.cap = None
        self.voice_thread = None
        self.voice_running = False
        self.status_label = None
        self.video_label = None
        self.header_icon = None
        self.content_frame = None
        self.sidebar = None
        self.window = window
        self.window.title("Hand Gesture Volume Control + Voice Control")
        self.window.geometry("1200x800")
        self.window.configure(bg="#f0f5f5")

        self.running = False
        self.model = None
        self.current_label = None
        self.data = []
        self.last_prediction = None
        self.prediction_count = 0
        self.theme = "light"

        self.setup_ui()

        try:
            if os.path.exists(resource_path('gesture_volume_model.joblib')):
                self.model = load(resource_path('gesture_volume_model.joblib'))
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def setup_ui(self):
        self.sidebar = tk.Frame(self.window, bg="#003366", width=200)
        self.sidebar.pack(side="left", fill="y")

        tk.Label(self.sidebar, text="Menu", font=("Arial", 16, "bold"), fg="white", bg="#003366").pack(pady=10)
        ttk.Button(self.sidebar, text="Home", command=self.show_home).pack(fill="x", padx=10, pady=5)
        ttk.Button(self.sidebar, text="Toggle Theme", command=self.toggle_theme).pack(fill="x", padx=10, pady=5)
        ttk.Button(self.sidebar, text="About", command=self.show_about).pack(fill="x", padx=10, pady=5)

        self.content_frame = tk.Frame(self.window, bg="#e6f2ff")
        self.content_frame.pack(side="left", fill="both", expand=True)
        self.build_main_ui()

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        bg_color = "#333333" if self.theme == "dark" else "#e6f2ff"
        text_color = "white" if self.theme == "dark" else "#003366"

        self.content_frame.configure(bg=bg_color)
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, (tk.Frame, tk.Label)):
                widget.configure(bg=bg_color, fg=text_color)

    def build_main_ui(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        top_frame = tk.Frame(self.content_frame, bg="#003366", height=80)
        top_frame.pack(fill="x")
        header_icon = Image.open(resource_path("icons/header.png")).resize((50, 50))
        self.header_icon = ImageTk.PhotoImage(header_icon)
        tk.Label(top_frame, image=self.header_icon, bg="#003366").pack(side="left", padx=20)
        tk.Label(top_frame, text="Hand Gesture Volume Control + Voice Control", font=("Arial", 20, "bold"), fg="white", bg="#003366").pack(side="left", pady=10)

        main_frame = tk.Frame(self.content_frame, bg="#e6f2ff", padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        self.video_label = tk.Label(main_frame, bg="#d9e6f2", bd=2, relief="sunken")
        self.video_label.pack(pady=10)
        self.status_label = tk.Label(main_frame, text="Gesture: None", font=("Arial", 18), bg="#e6f2ff", fg="#003366")
        self.status_label.pack(pady=10)

        btn_frame = tk.Frame(main_frame, bg="#e6f2ff")
        btn_frame.pack(pady=10)
        self.create_icon_button(btn_frame, "Volume Up", "volume_up", 0)
        self.create_icon_button(btn_frame, "Volume Down", "volume_down", 1)
        self.create_icon_button(btn_frame, "Mute", "mute", 2)
        self.create_icon_button(btn_frame, "Neutral", "neutral", 3)

        control_frame = tk.Frame(main_frame, bg="#e6f2ff")
        control_frame.pack(pady=10)
        self.create_icon_button(control_frame, "Start Collecting", "start", 0, action=self.start_capture)
        self.create_icon_button(control_frame, "Stop", "stop", 1, action=self.stop_capture)
        self.create_icon_button(control_frame, "Save Data", "save", 2, action=self.save_data)
        self.create_icon_button(control_frame, "Train Model", "train", 3, action=self.train_model)
        self.create_icon_button(control_frame, "Recognize", "recognize", 4, action=self.start_recognition)
        self.create_icon_button(control_frame, "Metrics", "metrics", 5, action=self.show_metrics)
        self.create_icon_button(control_frame, "Start Voice Control", "voice", 6, action=self.start_voice_control)
        self.create_icon_button(control_frame, "Stop Voice Control", "stop", 7, action=self.stop_voice_control)

        banner = Image.open(resource_path("icons/banner.png")).resize((2000, 200))
        self.banner_image = ImageTk.PhotoImage(banner)
        tk.Label(self.content_frame, image=self.banner_image, bg="#e6f2ff").pack(pady=(10, 0))

    def create_icon_button(self, frame, text, icon_name, column, action=None):
        icon_path = resource_path(f"icons/{icon_name}.png")
        icon = Image.open(icon_path).resize((50, 50))
        photo = ImageTk.PhotoImage(icon)
        btn = tk.Button(frame, text=text, image=photo, compound="top", font=("Arial", 10),
                        bg="#cce5ff", fg="black", relief="raised", bd=2,
                        command=lambda: action() if action else self.set_label(icon_name))
        btn.image = photo
        btn.grid(row=0, column=column, padx=10, pady=10)

    def set_label(self, label):
        self.current_label = label
        self.status_label.config(text=f"Current Label: {label}")

    def start_capture(self):
        if self.running:
            messagebox.showinfo("Info", "Already capturing.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the webcam.")
            return
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def stop_capture(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        self.status_label.config(text="Gesture: None")

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    features = extract_features(hand_landmarks.landmark, w, h)
                    if self.current_label:
                        self.data.append(features + [self.current_label])

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def save_data(self):
        if not self.data:
            messagebox.showwarning("No Data", "No data to save.")
            return
        feature_cols = [f"f{i}" for i in range(len(self.data[0]) - 1)] + ['label']
        df = pd.DataFrame(self.data, columns=feature_cols)
        df.to_csv(resource_path('gesture_data.csv'), index=False)
        messagebox.showinfo("Saved", "Data saved to gesture_data.csv")

    def train_model(self):
        try:
            df = pd.read_csv(resource_path('gesture_data.csv'))
            feature_cols = [col for col in df.columns if col != 'label']
            X = df[feature_cols]
            y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            model.fit(X_train, y_train)
            dump(model, resource_path('gesture_volume_model.joblib'), compress=3)
            self.model = model

            self.y_pred = model.predict(X_test)
            self.y_test = y_test
            self.labels = sorted(df['label'].unique())

            acc = model.score(X_test, y_test)
            report = classification_report(y_test, self.y_pred)
            print(report)

            messagebox.showinfo("Model Trained", f"Accuracy: {acc * 100:.2f}%\n\n{report}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_metrics(self):
        if self.y_test is None or self.y_pred is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        cm = confusion_matrix(self.y_test, self.y_pred, labels=self.labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        png_path = resource_path(f"confusion_matrix_{timestamp}.png")
        pdf_path = resource_path(f"confusion_matrix_{timestamp}.pdf")

        plt.savefig(png_path)

        with PdfPages(pdf_path) as pdf:
            pdf.savefig()
            plt.close()

            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
            ax.axis('off')
            report = classification_report(self.y_test, self.y_pred, labels=self.labels)
            ax.text(0.01, 0.99, report, fontsize=10, ha='left', va='top', wrap=True, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)

        plt.show()
        messagebox.showinfo("Metrics Saved", f"Saved as:\n{png_path}\n{pdf_path}")

    def start_recognition(self):
        if self.model is None:
            messagebox.showerror("No Model", "Train the model first.")
            return
        if self.running:
            messagebox.showinfo("Info", "Recognition already running.")
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the webcam.")
            return
        self.running = True
        self.thread = threading.Thread(target=self.recognition_loop, daemon=True)
        self.thread.start()

    def recognition_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            gesture = "None"
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    features = extract_features(hand_landmarks.landmark, w, h)
                    prediction = self.model.predict([features])[0]

                    if prediction == self.last_prediction:
                        self.prediction_count += 1
                        if self.prediction_count > 5:
                            gesture = prediction
                            if prediction == 'volume_up':
                                pyautogui.press("volumeup")
                            elif prediction == 'volume_down':
                                pyautogui.press("volumedown")
                            elif prediction == 'mute':
                                pyautogui.press("volumemute")
                    else:
                        self.prediction_count = 0

                    self.last_prediction = prediction

            self.status_label.config(text=f"Gesture: {gesture}")
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    # ===== Voice Control methods =====
    def voice_loop(self):
        pythoncom.CoInitialize()
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)

        while self.voice_running:
            with mic as source:
                try:
                    audio = recognizer.listen(source, timeout=5)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"Voice Command: {command}")

                    if "volume up" in command:
                        pyautogui.press("volumeup")
                    elif "volume down" in command:
                        pyautogui.press("volumedown")
                    elif "mute" in command:
                        pyautogui.press("volumemute")
                    elif "unmute" in command:
                        pyautogui.press("volumemute")  # toggle mute off

                except sr.UnknownValueError:
                    continue
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Voice error: {e}")

    def start_voice_control(self):
        if self.voice_running:
            messagebox.showinfo("Info", "Voice control already running.")
            return
        self.voice_running = True
        self.voice_thread = threading.Thread(target=self.voice_loop, daemon=True)
        self.voice_thread.start()
        messagebox.showinfo("Voice Control", "Voice control started. Say 'volume up', 'volume down', or 'mute'.")

    def adjust_volume(self, action):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current = volume.GetMasterVolumeLevelScalar()

        if action == "up":
            volume.SetMasterVolumeLevelScalar(min(current + 0.1, 1.0), None)
        elif action == "down":
            volume.SetMasterVolumeLevelScalar(max(current - 0.1, 0.0), None)
        elif action == "mute":
            volume.SetMute(1, None)
        elif action == "unmute":
            volume.SetMute(0, None)
        if prediction == 'volume_up':
            self.adjust_volume("up")
        elif prediction == 'volume_down':
            self.adjust_volume("down")
        elif prediction == 'mute':
            self.adjust_volume("mute")

    def stop_voice_control(self):
        if not self.voice_running:
            messagebox.showinfo("Info", "Voice control is not running.")
            return
        self.voice_running = False
        messagebox.showinfo("Voice Control", "Voice control stopped.")

    def show_home(self):
        self.build_main_ui()

    def show_about(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        tk.Label(self.content_frame, text="About This App", font=("Arial", 24, "bold"), bg="#e6f2ff", fg="#003366").pack(pady=20)

        about_text = (
            "\U0001F3AF Final Year Project: Hand Gesture Volume Control + Voice Control\n\n"
            "\U0001F9D1‍\U0001F4BB Members and Developers:\n MUTABAZI Samuel\n"
            "\U0001F4C5 Year: 2025\n\n"
            "\U0001F6E0️ Technologies Used:\n"
            " - Python\n - OpenCV\n - MediaPipe\n - Tkinter (GUI)\n - RandomForest\n - pyautogui\n - SpeechRecognition\n\n"
            "\U0001F4DA Purpose:\nControl PC volume with hand gestures and voice commands.\n"
            "\U0001F64F Thanks to our supervisors:\n 1. Dr KURADUSENGE Martin\n 2.Dr James RWIGEMA\n"
            "\U0001F4E9 Contact:\nsamumutabazi@gmail.com"
        )

        tk.Label(
            self.content_frame,
            text=about_text,
            justify="left",
            font=("Arial", 12),
            bg="#e6f2ff",
            fg="#003366",
            padx=20,
            anchor="w"
        ).pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()
