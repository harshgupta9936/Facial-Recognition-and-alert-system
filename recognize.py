import cv2
import numpy as np
from insightface.app import FaceAnalysis
from collections import deque
import pickle
import time
from alert_async import send_alert_async
import os
from datetime import datetime
from config import SIMILARITY_THRESHOLD, ALERT_COOLDOWN
from reid_db import ReIDDatabase


FRAME_SKIP = 2
UNKNOWN_TIME_THRESHOLD = 5
SMOOTH_WINDOW = 5
MATCH_IOU_THRESHOLD = 0.3


reid_db = ReIDDatabase(threshold=0.65)

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = np.array(data["encodings"])
known_names = np.array(data["names"])

known_encodings = known_encodings / np.linalg.norm(
    known_encodings, axis=1, keepdims=True
)

print(f"[INFO] Loaded {len(known_encodings)} encodings")


def norm(x):
    return x / (np.linalg.norm(x) + 1e-12)

def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])

    return inter / (areaA + areaB - inter + 1e-5)

def recognize_face(embedding):
    emb = norm(embedding)
    sims = np.dot(known_encodings, emb)

    idx = int(np.argmax(sims))
    score = float(sims[idx])
    name = known_names[idx]

    if score < SIMILARITY_THRESHOLD:
        return "Unknown", score

    return name, score


next_id = 0
tracks = {}
unknown_start_time = {}
last_alert_time = 0

def assign_tracks(faces, now):
    global next_id, tracks

    assigned = []
    used = set()

    for fd in faces:
        best_id, best_iou = None, 0

        for tid, t in tracks.items():
            if tid in used:
                continue

            iou = compute_iou(t["bbox"], fd["bbox"])
            if iou > best_iou:
                best_iou, best_id = iou, tid

        if best_id is not None and best_iou > MATCH_IOU_THRESHOLD:
            tracks[best_id]["bbox"] = fd["bbox"]
            tracks[best_id]["last_seen"] = now
            used.add(best_id)
            assigned.append((best_id, fd))
        else:
            tid = next_id
            next_id += 1
            tracks[tid] = {
                "bbox": fd["bbox"],
                "last_seen": now,
                "history": deque(maxlen=SMOOTH_WINDOW)
            }
            used.add(tid)
            assigned.append((tid, fd))

    for tid in list(tracks.keys()):
        if now - tracks[tid]["last_seen"] > 1:
            del tracks[tid]

    return assigned

def smooth_name(tid, name):
    if name.startswith("unknown"):
        return name

    tracks[tid]["history"].append(name)
    counts = {}

    for n in tracks[tid]["history"]:
        counts[n] = counts.get(n, 0) + 1

    return max(counts, key=counts.get)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    face_data = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        face_data.append({
            "bbox": (x1, y1, x2, y2),
            "embedding": f.embedding
        })

    now = time.time()
    assigned = assign_tracks(face_data, now)

    for tid, fd in assigned:
        emb = fd["embedding"]

        name, score = recognize_face(emb)

        
        # ReID (Re-Identification System)
        if name == "Unknown":
            reid_name, reid_score = reid_db.match(emb)

            if reid_name is None:
                reid_name = reid_db.add_unknown(emb)
            else:
                reid_db.update(reid_name, emb)

            name = reid_name
            score = reid_score if reid_score is not None else 0.0

        else:
            name = smooth_name(tid, name)



        # Alert + Save IMAGE + VIDEO
        if name.startswith("unknown"):
            if name not in unknown_start_time:
                unknown_start_time[name] = now

            duration = now - unknown_start_time[name]

            if duration > UNKNOWN_TIME_THRESHOLD:
                if now - last_alert_time > ALERT_COOLDOWN:

                    os.makedirs("intruders", exist_ok=True)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                    image_path = f"intruders/intruder_{ts}.jpg"
                    video_path = f"intruders/intruder_{ts}.mp4"

        
                    # Save IMAGE
        
                    annotated = frame.copy()
                    x1, y1, x2, y2 = fd["bbox"]

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    cv2.putText(
                        annotated,
                        "INTRUDER",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(
                        annotated,
                        timestamp,
                        (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.imwrite(image_path, annotated)

        
                    # Save VIDEO
        
                    fps = 20
                    duration_sec = 5
                    frame_count = int(fps * duration_sec)

                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

                    print("[INFO] Recording intruder clip...")

                    for _ in range(frame_count):
                        ret, clip_frame = cap.read()
                        if not ret:
                            break

                        annotated_clip = clip_frame.copy()

                        cv2.rectangle(annotated_clip, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        cv2.putText(
                            annotated_clip,
                            "INTRUDER",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(
                            annotated_clip,
                            timestamp,
                            (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )

                        out.write(annotated_clip)

                    out.release()

        
                    # Send BOTH
        
                    send_alert_async(
                        subject="Intruder Alert",
                        body=f"{name} detected for {int(duration)} sec",
                        attachments=[image_path, video_path]
                    )

                    last_alert_time = now
                    unknown_start_time[name] = now
        else:
            if name in unknown_start_time:
                del unknown_start_time[name]


        x1, y1, x2, y2 = fd["bbox"]
        color = (0,255,0) if not name.startswith("unknown") else (0,0,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{name} ({score:.2f})",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Maintenanc
    if int(now) % 5 == 0:
        reid_db.save()

    if int(now) % 10 == 0:
        reid_db.merge_similar()

    # cv2.imshow("Face Recognition", frame)
    display_frame = cv2.resize(frame, (900, 550))  # width, height
    cv2.imshow("Face Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
