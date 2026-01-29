import cv2
import os
import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
from datetime import datetime
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBED_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
ATTENDANCE_CSV = os.path.join(DATA_DIR, "attendance.csv")

# ---------------- LOAD MODELS ----------------
detector = MTCNN()
embedder = FaceNet()

# ---------------- LOAD DATABASE ----------------
if not os.path.exists(EMBED_PATH):
    print("❌ No registered users found!")
    exit()

with open(EMBED_PATH, "rb") as f:
    db = pickle.load(f)

# ---------------- UTILS ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def average_similarity(test_emb, known_embs):
    if isinstance(known_embs, list):
        scores = [cosine_similarity(test_emb, e) for e in known_embs]
        return np.mean(scores)
    else:
        return cosine_similarity(test_emb, known_embs)

RECOGNITION_THRESHOLD = 0.92  # strict

attendance = {
    "current_user": None,
    "punched_in": False,
    "punched_out": False
}

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- TKINTER WINDOW ----------------
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("800x600")

label = Label(root)
label.pack()

status_label = Label(root, text="Status: Waiting...", font=("Helvetica", 14))
status_label.pack(pady=10)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) != 1:
        cv2.putText(frame, "Show ONLY ONE face", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        status_label.config(text="Status: Waiting for a single face")
    else:
        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)
        face = rgb[y:y+h, x:x+w]
        if face.size != 0:
            face_resized = cv2.resize(face, (160, 160))
            face_norm = face_resized.astype("float32")
            face_norm = (face_norm - face_norm.mean()) / (face_norm.std() + 1e-8)
            embedding = embedder.embeddings(np.expand_dims(face_norm, axis=0))[0]

            best_name = None
            best_score = 0
            for name, emb_list in db.items():
                score = average_similarity(embedding, emb_list)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= RECOGNITION_THRESHOLD:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, best_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                status_label.config(text=f"Status: Recognized {best_name}")

                # Punch system
                if attendance["current_user"] is None:
                    attendance["current_user"] = best_name
                elif attendance["current_user"] != best_name:
                    status_label.config(text="Status: Identity changed!")
                    cv2.putText(frame, "❌ IDENTITY CHANGED", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if not attendance["punched_in"]:
                    attendance["punched_in"] = True
                    with open(ATTENDANCE_CSV, "a") as f:
                        f.write(f"{best_name},{datetime.now()},Punch-In\n")
                elif attendance["punched_in"] and not attendance["punched_out"]:
                    attendance["punched_out"] = True
                    with open(ATTENDANCE_CSV, "a") as f:
                        f.write(f"{best_name},{datetime.now()},Punch-Out\n")
            else:
                cv2.putText(frame, "❌ UNKNOWN PERSON", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                status_label.config(text="Status: Unknown person")

    # Convert to ImageTk
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)

# ---------------- START ----------------
update_frame()
root.mainloop()
cap.release()
