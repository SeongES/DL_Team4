# DL_Team4

## Project Title  
**Assistance in Car Roads for Visual-Impaired People**  
> A deep learning system that detects road objects, estimates their distance, and provides real-time audio alerts to assist visually-impaired individuals in navigating car roads.

---

## 🔧 Setup

### 1. Dataset Download

We use the **BDD100K dataset** for object detection.

- 📦 [Download BDD100K Dataset from Kaggle](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)

### 2. Pretrained Models and Annotations

Please download the following files and place them in the appropriate directories as specified in the notebooks:

- 📄 [`gt_depths.npz`](https://drive.google.com/file/d/1OpMHwBZrsHzO-lOheG2lbpUGAmow-L_-/view?usp=sharing) — Precomputed ground truth depths
- 📄 [`bdd100k_labels_images_train_coco.json`](https://drive.google.com/file/d/1cqueQXoroEzOQc874oo6jjb5nTvermVx/view?usp=sharing) — BDD100K annotations in COCO format

---

## 📁 File Descriptions

### 🔹 `train.ipynb`  
- YOLOv11n object detection model training notebook  
- Finetunes YOLOv11n on the BDD100K dataset

### 🔹 `depth_estimation.ipynb`  
- Performs monocular depth estimation using **LiteMono**  
- Based on the official [LiteMono GitHub repository](https://github.com/noahzn/Lite-Mono)

### 🔹 `demo.py`  
- Real-time system integrating:
  - Object Detection (YOLO)
  - Depth Estimation (LiteMono)
  - Text-to-Speech (TTS) alerts  
- Designed to assist visually-impaired users by announcing detected objects and their relative distance

