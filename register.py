import cv2
import os
import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBED_PATH = os.path.join(DATA_DIR, "embeddings.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
detector = MTCNN()
embedder = FaceNet()

# ---------------- UTILS ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ---------------- LOAD DATABASE ----------------
if os.path.exists(EMBED_PATH):
    with open(EMBED_PATH, "rb") as f:
        db = pickle.load(f)
else:
    db = {}

# ---------------- USER INPUT ----------------
name = input("Enter user name: ").strip()
if not name:
    print("❌ Name cannot be empty")
    exit()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
embeddings = []

print("📸 Look at the camera... Capturing face samples")

while len(embeddings) < 10:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    # Only one face allowed
    if len(faces) != 1:
        cv2.putText(frame, "Show only ONE face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Register Face", frame)
        cv2.waitKey(1)
        continue

    x, y, w, h = faces[0]['box']
    x, y = abs(x), abs(y)

    face = rgb[y:y+h, x:x+w]
    if face.size == 0:
        continue

    face = cv2.resize(face, (160, 160))
    face = face.astype("float32")
    mean, std = face.mean(), face.std()
    face = (face - mean) / (std + 1e-8)

    embedding = embedder.embeddings(np.expand_dims(face, axis=0))[0]
    embeddings.append(embedding)

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"Samples: {len(embeddings)}/10",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- SAFETY CHECK ----------------
if len(embeddings) == 0:
    print("❌ No face captured")
    exit()

new_embedding = np.mean(embeddings, axis=0)

# ---------------- DUPLICATE CHECK ----------------
THRESHOLD = 0.75

for existing_name, existing_emb in db.items():
    similarity = cosine_similarity(new_embedding, existing_emb)
    if similarity >= THRESHOLD:
        print("❌ Face already registered!")
        print(f"Registered as: {existing_name}")
        exit()

# ---------------- SAVE NEW FACE ----------------
db[name] = new_embedding

with open(EMBED_PATH, "wb") as f:
    pickle.dump(db, f)

print(f"✅ Face registered successfully for {name}")



