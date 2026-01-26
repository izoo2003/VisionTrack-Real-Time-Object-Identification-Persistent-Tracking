# VisionTrack: Real-Time Object Identification & Persistent Tracking

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)
![License](https://img.shields.io/badge/License-AGPL_3.0-red.svg)

VisionTrack is a high-performance computer vision system that integrates **YOLOv8** for real-time object detection with **Deep SORT** for persistent multi-object tracking. By utilizing Kalman filtering and deep learning embeddings, the system identifies objects, maintains unique IDs through occlusions, and monitors live counts and FPS.

---

## 🚀 Key Features
* **Real-Time Detection:** Powered by YOLOv8s for an optimal balance between speed and accuracy.
* **Persistent Tracking:** Deep SORT implementation ensures objects keep their unique IDs even when briefly blocked or moved.
* **Intelligent Filtering:** Optimized to ignore background "noise" (like misidentified person detections) and focus on specific objects.
* **Live Analytics:** Real-time display of FPS (Frames Per Second) and a unique object counter.
* **Visual Interface:** Clean UI with bounding boxes, object labels, and tracking IDs.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/izoo2003/VisionTrack-Real-Time-Object-Identification-Persistent-Tracking.git](https://github.com/izoo2003/VisionTrack-Real-Time-Object-Identification-Persistent-Tracking.git)
cd VisionTrack-Real-Time-Object-Identification-Persistent-Tracking

### 2. Create An Virtual Enviroment

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

### 3. Install The Dependencies
pip install -r requirements.txt
ultralytics (YOLOv8 framework)
opencv-python (Image processing & UI)
deep-sort-realtime (Tracking algorithm)
