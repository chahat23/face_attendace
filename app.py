from flask import Flask, render_template, Response
import cv2
import os
import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
from datetime import datetime

app = Flask(__name__)

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

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if len(faces) != 1:
            cv2.putText(frame, "Show only ONE face", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            x, y, w, h = faces[0]['box']
            x, y = abs(x), abs(y)
            face = rgb[y:y+h, x:x+w]
            if face.size != 0:
                face_resized = cv2.resize(face, (160, 160))
                face_norm = face_resized.astype("float32")
                face_norm = (face_norm - face_norm.mean()) / (face_norm.std() + 1e-8)
                embedding = embedder.embeddings(np.expand_dims(face_norm, axis=0))[0]

                # Matching
                best_name = None
                best_score = 0
                for name, emb_list in db.items():
                    score = average_similarity(embedding, emb_list)
                    if score > best_score:
                        best_score = score
                        best_name = name

                if best_score >= RECOGNITION_THRESHOLD:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{best_name}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Punch system
                    if attendance["current_user"] is None:
                        attendance["current_user"] = best_name
                    elif attendance["current_user"] != best_name:
                        cv2.putText(frame, "❌ IDENTITY CHANGED", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        continue

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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
