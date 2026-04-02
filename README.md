# 🚀 Real-Time Face Recognition & Intruder Alert System (ReID Enabled)

A robust **real-time face recognition and surveillance system** with intelligent intruder detection, Re-Identification (ReID), and automated email alerts with visual evidence.

Designed to run efficiently on **CPU-based systems**, making it practical for real-world deployment without requiring high-end GPUs.

---

## 📌 Features

### 👤 Face Recognition

* Uses **InsightFace (buffalo_l model)** for high-quality embeddings
* Cosine similarity-based matching
* Configurable recognition threshold

### 🔁 Re-Identification (ReID)

* Tracks unknown individuals across frames
* Assigns persistent identities (`unknown_0`, `unknown_1`, etc.)
* Dynamically merges similar identities to reduce duplicates

### 🎯 Smart Tracking System

* IOU-based tracking for consistent identity assignment
* Temporal smoothing to avoid flickering predictions
* Stable identity labeling in real-time video

### 🚨 Intruder Detection

* Detects unknown individuals based on **time persistence**
* Avoids false positives using threshold logic
* Cooldown mechanism prevents alert spamming

### 📸 Evidence Collection

* Captures:

  * Annotated image
  * 5-second annotated video clip

### 📧 Asynchronous Email Alerts

* Sends alerts without blocking the main pipeline
* Supports multiple attachments (image + video)
* Uses secure SMTP (Gmail App Password)

---

## 🏗️ Project Structure

```
.
├── dataset/                # Known faces (organized by person name)
├── intruders/              # Saved intruder alerts
├── encodings.pkl           # Face embeddings database
├── reid_db.pkl             # Unknown identity database
│
├── encode_faces.py         # Generate embeddings from dataset
├── recognize.py            # Main real-time recognition system
├── reid_db.py              # ReID identity management
├── alert_async.py          # Email alert system (async)
├── config.py               # Configuration file
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/face-recognition-alert.git
cd face-recognition-alert
```

### 2️⃣ Install Dependencies

```bash
pip install opencv-python numpy insightface
```

---

## 🧑‍💻 Setup

### 📂 1. Prepare Dataset

Organize known faces like this:

```
dataset/
 ├── person1/
 │    ├── img1.jpg
 │    ├── img2.jpg
 ├── person2/
 │    ├── img1.jpg
```

---

### 🧠 2. Encode Faces

```bash
python encode_faces.py
```

This will generate:

```
encodings.pkl
```

---

### 📧 3. Configure Email Alerts

Edit `config.py`:

```python
EMAIL = "your_email@gmail.com"
PASSWORD = "your_app_password"
TO_EMAIL = "receiver_email@gmail.com"
```

⚠️ **Important:**

* Use a **Google App Password**, not your main password
* Enable 2FA before generating the app password

---

## ▶️ Running the System

```bash
python recognize.py
```

Press **ESC** to exit.

---

## 🔄 How It Works

1. Webcam captures live video
2. Faces are detected using InsightFace
3. Embeddings are extracted and normalized
4. Compared against known encodings
5. If unknown:

   * Passed to ReID system
   * Assigned persistent identity
6. If unknown persists:

   * Intruder alert triggered
   * Image + video saved
   * Email sent asynchronously
7. Background tasks:

   * ReID database saved periodically
   * Similar identities merged

---

## ⚙️ Configuration Parameters

Modify in `config.py`:

| Parameter                 | Description                  |
| ------------------------- | ---------------------------- |
| `SIMILARITY_THRESHOLD`    | Recognition strictness       |
| `ALERT_COOLDOWN`          | Minimum time between alerts  |
| `UNKNOWN_FRAME_THRESHOLD` | Intruder trigger sensitivity |
| `VIDEO_SOURCE`            | Webcam index                 |

---

## ⚡ Performance Notes

* Optimized for **CPU execution** (`ctx_id = -1`)
* Works on systems like **GTX 1650 Max-Q / no GPU**
* Adjustable thresholds for speed vs accuracy

---

## 🧪 Tech Stack

* **Python**
* **OpenCV**
* **InsightFace**
* **NumPy**
* **SMTP (Email Alerts)**

---

## 🔮 Future Improvements

* GPU acceleration support
* Web dashboard (live monitoring)
* SMS / push notifications
* Multi-camera support
* Cloud deployment (AWS / GCP)

---


