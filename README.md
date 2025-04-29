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

