import cv2
import os
import pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
from datetime import datetime

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

print(f"✅ Registered users: {list(db.keys())}")

# ---------------- UTILS ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def max_similarity(test_emb, known_embs):
    if isinstance(known_embs, list):
        return max([cosine_similarity(test_emb, e) for e in known_embs])
    else:
        return cosine_similarity(test_emb, known_embs)

# ---------------- THRESHOLD ----------------
RECOGNITION_THRESHOLD = 0.85  # Much stricter threshold (increase to 0.90 if still issues)

# ---------------- ATTENDANCE STATE ----------------
attendance = {
    "current_user": None,
    "punched_in": False,
    "punched_out": False,
    "punch_in_time": None,
    "consecutive_frames": 0,  # Track consecutive recognitions
    "last_best_name": None
}

# ---------------- LIVENESS STATE ----------------
prev_center = None
movement_count = 0
REQUIRED_MOVEMENT = 10  # minimum movement in pixels

# ---------------- RECOGNITION STATE ----------------
CONSECUTIVE_FRAMES_REQUIRED = 15  # Increased from 10 to 15 for more stability
MIN_PUNCH_INTERVAL = 5  # Minimum 5 seconds between punch-in and punch-out
RECOGNITION_HISTORY = []  # Track last N recognitions
HISTORY_SIZE = 20

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera!")
    exit()

print("📸 Secure Face Recognition Started")
print("=" * 50)
print("INSTRUCTIONS:")
print("- Face verified over 15 consecutive frames (strict)")
print("- Threshold: 0.85 (85% similarity required)")
print("- Punch-In happens automatically after verification")
print("- Press 'P' to Punch-Out (after 5 seconds)")
print("- Press 'ESC' to exit")
print("=" * 50)
print("WATCH CONSOLE for detailed matching scores...")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    # ---------- BLOCK if no faces ----------
    if len(faces) == 0:
        cv2.putText(frame, "Show face", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # ---------- BLOCK if multiple faces ----------
    if len(faces) > 1:
        cv2.putText(frame, "❌ Multiple faces detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # ---------- PROCESS SINGLE FACE ----------
    x, y, w, h = faces[0]['box']
    x, y = abs(x), abs(y)
    face = rgb[y:y+h, x:x+w]
    if face.size == 0:
        continue

    # ---------- HEAD MOVEMENT CHECK (anti-spoof) ----------
    center = (x + w//2, y + h//2)
    if prev_center is not None:
        dist = np.linalg.norm(np.array(center) - np.array(prev_center))
        if dist > REQUIRED_MOVEMENT:
            movement_count += 1
    prev_center = center

    if movement_count < 1:  # Require minimal head movement
        cv2.putText(frame, "❌ Move head slightly", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Recognition", frame)
        cv2.waitKey(1)
        continue

    # ---------- FACE EMBEDDING ----------
    face_resized = cv2.resize(face, (160, 160))
    face_norm = face_resized.astype("float32")
    face_norm = (face_norm - face_norm.mean()) / (face_norm.std() + 1e-8)
    embedding = embedder.embeddings(np.expand_dims(face_norm, axis=0))[0]

    # ---------- MATCHING (Strict validation) ----------
    best_name = "UNKNOWN"
    best_score = 0
    all_scores = {}  # Track all similarity scores
    
    for name, emb_list in db.items():
        score = max_similarity(embedding, emb_list)
        all_scores[name] = score
        # Only consider it a match if it meets the threshold
        if score >= RECOGNITION_THRESHOLD and score > best_score:
            best_score = score
            best_name = name
    
    # Additional validation: Check the gap between best and second-best
    sorted_scores = sorted(all_scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        score_gap = sorted_scores[0] - sorted_scores[1]
        # If top two scores are too close, it's ambiguous - mark as UNKNOWN
        if score_gap < 0.05:  # Require at least 5% difference
            best_name = "UNKNOWN"
            print(f"⚠️ Ambiguous match - scores too close: {sorted_scores[0]:.4f} vs {sorted_scores[1]:.4f}")
    
    # Debug: print all scores
    print(f"Frame scores: {', '.join([f'{name}: {score:.4f}' for name, score in all_scores.items()])}")
    if best_name != "UNKNOWN":
        print(f"✓ Best Match: {best_name} with score: {best_score:.4f}")
    else:
        print(f"✗ No valid match (best score: {max(all_scores.values()):.4f}, threshold: {RECOGNITION_THRESHOLD})")
    
    # Track recognition history
    RECOGNITION_HISTORY.append(best_name)
    if len(RECOGNITION_HISTORY) > HISTORY_SIZE:
        RECOGNITION_HISTORY.pop(0)
    
    # ---------- CONSECUTIVE FRAME VERIFICATION ----------
    if best_name != "UNKNOWN":
        if attendance["last_best_name"] == best_name:
            attendance["consecutive_frames"] += 1
        else:
            # Different person or first detection
            attendance["consecutive_frames"] = 1
            attendance["last_best_name"] = best_name
    else:
        attendance["consecutive_frames"] = 0
        attendance["last_best_name"] = None

    # ---------- LOCK IDENTITY (only after consecutive frames) ----------
    confirmed_name = "UNKNOWN"
    
    if best_name != "UNKNOWN" and attendance["consecutive_frames"] >= CONSECUTIVE_FRAMES_REQUIRED:
        # Extra validation: Check consistency in recent history
        recent_history = RECOGNITION_HISTORY[-CONSECUTIVE_FRAMES_REQUIRED:]
        name_counts = {name: recent_history.count(name) for name in set(recent_history)}
        most_common_name = max(name_counts, key=name_counts.get)
        consistency = name_counts.get(most_common_name, 0) / len(recent_history)
        
        # Require at least 80% consistency
        if consistency >= 0.8 and most_common_name == best_name:
            confirmed_name = best_name
            
            if attendance["current_user"] is None:
                # First recognized user - lock to them
                attendance["current_user"] = best_name
                print(f"✅ Identity locked: {best_name} (consistency: {consistency*100:.1f}%)")
            elif attendance["current_user"] != best_name:
                # Different registered user detected - treat as UNKNOWN
                confirmed_name = "UNKNOWN"
                attendance["consecutive_frames"] = 0
                print(f"⚠️ Different user detected: {best_name} (locked to: {attendance['current_user']})")
        else:
            print(f"⚠️ Insufficient consistency: {consistency*100:.1f}% (need 80%)")
    
    # Show verification progress
    if best_name != "UNKNOWN" and attendance["consecutive_frames"] < CONSECUTIVE_FRAMES_REQUIRED:
        cv2.putText(frame, f"Verifying {best_name}: {attendance['consecutive_frames']}/{CONSECUTIVE_FRAMES_REQUIRED}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    # Use confirmed_name for display and attendance
    display_name = confirmed_name if confirmed_name != "UNKNOWN" else best_name

    # ---------- SINGLE PUNCH SYSTEM ----------
    if confirmed_name != "UNKNOWN" and attendance["consecutive_frames"] >= CONSECUTIVE_FRAMES_REQUIRED:
        current_time = datetime.now()
        
        if not attendance["punched_in"]:
            attendance["punched_in"] = True
            attendance["punch_in_time"] = current_time
            with open(ATTENDANCE_CSV, "a") as f:
                f.write(f"{confirmed_name},{current_time},Punch-In\n")
            print(f"🟢 Punch-In: {confirmed_name}")

        elif attendance["punched_in"] and not attendance["punched_out"]:
            # Check if enough time has passed since punch-in
            time_diff = (current_time - attendance["punch_in_time"]).total_seconds()
            if time_diff >= MIN_PUNCH_INTERVAL:
                # Show message that user can punch out by pressing 'P' key
                cv2.putText(frame, "Press 'P' to Punch-Out", (20, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                remaining = int(MIN_PUNCH_INTERVAL - time_diff)
                cv2.putText(frame, f"Wait {remaining}s to punch-out", (20, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # ---------- DISPLAY ----------
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if display_name != "UNKNOWN" else (0, 0, 255), 2)
    cv2.putText(frame, display_name, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if display_name != "UNKNOWN" else (0, 0, 255), 2)

    cv2.imshow("Recognition", frame)
    
    # Handle keypresses
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('p') or key == ord('P'):  # P to punch-out
        if attendance["punched_in"] and not attendance["punched_out"] and confirmed_name != "UNKNOWN":
            current_time = datetime.now()
            time_diff = (current_time - attendance["punch_in_time"]).total_seconds()
            if time_diff >= MIN_PUNCH_INTERVAL:
                attendance["punched_out"] = True
                with open(ATTENDANCE_CSV, "a") as f:
                    f.write(f"{confirmed_name},{current_time},Punch-Out\n")
                print(f"🔴 Punch-Out: {confirmed_name}")
            else:
                print(f"⚠️ Please wait {int(MIN_PUNCH_INTERVAL - time_diff)} more seconds")

cap.release()
cv2.destroyAllWindows()
print("🛑 Recognition stopped")
