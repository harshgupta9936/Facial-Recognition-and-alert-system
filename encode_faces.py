import cv2
import numpy as np
import os
import pickle
from insightface.app import FaceAnalysis

DATASET_PATH = "dataset"

# loading model (CPU- as i have no gpu)
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)

encodings = []
names = []

# processing the dataset here:
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing: {person_name}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not read {img_path}")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"[WARNING] No face found in {img_name}")
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        encodings.append(emb)
        names.append(person_name)

        print(f"[OK] {img_name}")

print(f"\n[INFO] Total encodings: {len(encodings)}")

with open("encodings.pkl", "wb") as f:
    pickle.dump({
        "encodings": encodings,
        "names": names
    }, f)

print("[SUCCESS] encodings.pkl saved")