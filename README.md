# 🚗 Computer Vision for Autonomous Driving

This project implements a complete computer vision pipeline for autonomous vehicles using **YOLOv8** trained on the **KITTI dataset**, evaluated in the **CARLA simulator**. It spans data preparation, model training, and deployment in a 3D virtual environment to detect cars, pedestrians, cyclists, and more in real-time.

---

## 📦 Features

- 📚 KITTI dataset preprocessing & annotation conversion (YOLO format)
- 🧠 YOLOv8 training with 8 object classes
- 📈 Evaluation with mAP, F1, precision, recall, confusion matrix & ROC curves
- 🚙 Simulation in CARLA with ego vehicle, traffic, and pedestrian actors
- 🎥 Real-time detection with RGB camera feed rendered as video

---

## 🧰 Tech Stack

- **Python**, **PyTorch**, **OpenCV**, **NumPy**
- **Ultralytics YOLOv8** (object detection)
- **CARLA Simulator 0.9.14** (simulation environment)
- **KITTI Dataset** (autonomous driving benchmark)

---

## 🗂️ Project Structure
```
/
├── Final.ipynb              # End-to-end training and inference notebook
├── carla_demo.py            # Script for CARLA simulation and deployment
├── kitti_dataset/           # Contains KITTI images, labels, and YOLO-formatted data
├── runs/
│   └── detect/              # YOLOv8 training outputs (weights, logs, results)
└── README.md                # Project overview (this file)

```


---

## 📊 Results Summary

- **mAP@50**: 0.90
- **F1 Score**: 0.978 (Micro), 0.969 (Macro)
- **Top Class Performance**: Car (F1 = 0.9964)
- **Simulated Video**: 600 ticks @ 20 FPS rendered via CARLA RGB camera

> 📽️ _Demo video available in presentation slides (not uploaded here due to size)._

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/syedabdulsamadahmedali/Computer-Vision.git
cd Computer-Vision
