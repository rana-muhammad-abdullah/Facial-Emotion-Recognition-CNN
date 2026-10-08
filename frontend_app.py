import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# CONFIGURATION
MODEL_PATH = 'emotion_recognition_model.keras'
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOJI_DIR = 'emojis/'

facial_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 
               4: "neutral", 5: "sad", 6: "surprise"}

class EmojiApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Emoji Detector")
        
        # Load AI Model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run main_train.py to train the model first.")
        self.model = load_model(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        
        # Setup UI
        self.video_label = tk.Label(window)
        self.video_label.pack(side="left", padx=10, pady=10)
        
        self.emoji_display = tk.Label(window)
        self.emoji_display.pack(side="right", padx=50)
        
        self.status_label = tk.Label(window, text="Detecting...", font=("Arial", 18))
        self.status_label.pack(side="bottom", pady=20)

        self.cap = None
        for i in range(5):  # Try camera indices 0 to 4
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                break
        if not self.cap or not self.cap.isOpened():
            self.status_label.config(text="Camera not found")
            return
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = np.expand_dims(np.expand_dims(roi, -1), 0)

                prediction = self.model.predict(roi, verbose=0)
                label = facial_dict[np.argmax(prediction)]
                
                self.status_label.config(text=f"Feeling: {label}")
                self.show_emoji(label)

            # Convert for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = self.imgtk
            self.video_label.configure(image=self.imgtk)
            
        self.window.after(10, self.update_frame)

    def show_emoji(self, emotion):
        path = os.path.join(EMOJI_DIR, f"{emotion.capitalize()}.png")
        if os.path.exists(path):
            img = Image.open(path).resize((200, 200))
            emoji_img = ImageTk.PhotoImage(img)
            self.emoji_display.configure(image=emoji_img)
            self.emoji_display.image = emoji_img

if __name__ == "__main__":
    root = tk.Tk()
    app = EmojiApp(root)
    root.mainloop()