🔹 Project Title

Real-Time Face Recognition with Intruder Alert System (ReID Enabled)

🔹 Overview

This project implements a real-time face recognition and surveillance system with:

Known face recognition (via pre-encoded embeddings)
Unknown person tracking using Re-Identification (ReID)
Intruder detection based on temporal persistence
Automatic email alerts with image + video evidence
Identity smoothing and tracking for stability

The system is optimized for CPU environments and works efficiently on mid-range hardware like GTX 1650 Max-Q.

🔹 Key Features
✅ Face Recognition
Uses InsightFace (buffalo_l model) for embeddings
Cosine similarity-based matching
Configurable similarity threshold
🔁 Re-Identification (ReID)
Tracks unknown individuals across frames
Assigns persistent IDs like unknown_0, unknown_1
Merges similar identities dynamically
🎯 Tracking System
IOU-based tracking for consistent identity assignment
Temporal smoothing using sliding window
Prevents flickering identities
🚨 Intruder Detection Logic
Triggers alert when:
Person is unknown
Visible for more than threshold time
Cooldown mechanism prevents spam alerts
📸 Evidence Capture
Saves:
Annotated image
5-second annotated video clip
📧 Async Email Alerts
Sends alerts without blocking main thread
Supports multiple attachments (image + video)


🔹 Project Structure
.
├── dataset/                # Known faces (organized by person name)
├── intruders/              # Saved alerts (images + videos)
├── encodings.pkl           # Precomputed face embeddings
├── reid_db.pkl             # Unknown identity database
│
├── encode_faces.py         # Dataset encoding script
├── recognize.py            # Main real-time system
├── reid_db.py              # ReID identity manager
├── alert_async.py          # Email alert system
├── config.py               # Configurations


🔹 Installation
1. Clone Repository
git clone <repo-url>
cd <repo>
2. Install Dependencies
pip install opencv-python numpy insightface


🔹 Setup
1. Prepare Dataset
dataset/
 ├── person1/
 │    ├── img1.jpg
 │    ├── img2.jpg
 ├── person2/
2. Encode Faces
python encode_faces.py
3. Configure Email

Edit config.py:

EMAIL = "your_email@gmail.com"
PASSWORD = "your_app_password"
TO_EMAIL = "receiver@gmail.com"
⚠️ Use Google App Password, not your main password.


🔹 Run the System
python recognize.py

      
🔹 How It Works (Pipeline)
Webcam feed captured
Faces detected + embeddings extracted
Compared with known encodings
If unknown:
Passed to ReID system
Assigned persistent identity
If unknown persists:
Image + video recorded
Email alert triggered
Background maintenance:
ReID DB saved periodically
Similar identities merged


🔹 Configuration Parameters
Parameter	Description
SIMILARITY_THRESHOLD	Recognition strictness
ALERT_COOLDOWN	Time between alerts
UNKNOWN_TIME_THRESHOLD	Intruder trigger delay
FRAME_SKIP	Performance optimization

🔹 Performance Notes
CPU-optimized (ctx_id = -1)
Works on low GPU systems
Adjustable thresholds for accuracy vs speed


🔹 Future Improvements
GPU acceleration support
Web dashboard for monitoring
SMS / push notifications
Multi-camera support
Face mask handling
