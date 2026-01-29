import cv2
import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
detector = MTCNN()
embedder = FaceNet()
THRESHOLD = 0.45

# ---------- Load DB ----------
with open("data/embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# ---------- Utilities ----------
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------- Input ----------
new_name = input("Enter NEW name: ").strip()
if not new_name:
    print("❌ Name cannot be empty")
    exit()

print("📸 Look at the camera to identify the face to rename")

cap = cv2.VideoCapture(0)
embedding = None

while embedding is None:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) != 1:
        cv2.putText(frame, "Show ONE face", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Rename Face", frame)
        cv2.waitKey(1)
        continue

    x,y,w,h = faces[0]['box']
    x,y = abs(x), abs(y)
    face = rgb[y:y+h, x:x+w]

    if face.size == 0:
        continue

    face = cv2.resize(face, (160,160))
    face = face.astype("float32")
    face = (face - face.mean()) / (face.std() + 1e-8)
    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)[0]

cap.release()
cv2.destroyAllWindows()

# ---------- Find Existing Face ----------
matched_name = None

for name, emb in db.items():
    if cosine_distance(emb, embedding) < THRESHOLD:
        matched_name = name
        break

if not matched_name:
    print("❌ Face not found in database")
    exit()

# ---------- Rename ----------
db[new_name] = db.pop(matched_name)

with open("data/embeddings.pkl", "wb") as f:
    pickle.dump(db, f)

print(f"✅ Renamed '{matched_name}' ➜ '{new_name}' successfully")
