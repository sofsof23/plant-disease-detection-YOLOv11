# 🌿 Plant Disease Detection – YOLOv11

A deep learning object detection project that uses **YOLOv11** to detect and classify plant diseases from leaf images, helping farmers and agronomists identify crop issues early.

> Trained on Kaggle using an NVIDIA Tesla T4 GPU.

---

## 📋 About

Plant diseases cause significant crop losses worldwide every year. Early and accurate detection is key to preventing large-scale agricultural damage. This project applies YOLOv11 to automatically detect and classify disease types directly from leaf images in real time.

The model was trained on the **PlantVillage** dataset and detects 8 disease/health classes across tomato, potato, and pepper plants.

---

## 🧠 Model

| Detail | Value |
|--------|-------|
| Architecture | YOLOv11 (Ultralytics) |
| Task | Object Detection & Classification |
| Dataset | PlantVillage |
| Classes | 8 (see below) |
| Weights file | `best.pt` |
| Training platform | Kaggle (NVIDIA Tesla T4 GPU) |
| Epochs | 50 |
| Image size | 640×640 |

---

## 🌱 Classes

| ID | Class |
|----|-------|
| 0 | Tomato – Early Blight |
| 1 | Tomato – Late Blight |
| 2 | Tomato – Healthy |
| 3 | Potato – Early Blight |
| 4 | Potato – Late Blight |
| 5 | Potato – Healthy |
| 6 | Pepper – Bacterial Spot |
| 7 | Pepper – Healthy |

---

## 🗂️ Project Structure

```
plant-disease-detection-yolov11/
├── best.pt                                        # Trained model weights
├── training-plant-disease-detection-yolov11.ipynb # Training notebook (Kaggle)
├── predict.py                                     # Run inference on new images
├── data.yaml                                      # Dataset configuration
├── test/
│   ├── images/                                    # Test leaf images
│   └── labels/                                    # Ground truth labels (YOLO format)
└── README.md
```

---

## 🛠️ Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00BFFF?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/sofsof23/plant-disease-detection-yolov11.git
cd plant-disease-detection-yolov11
```

### 2. Install dependencies
```bash
pip install ultralytics opencv-python matplotlib
```

### 3. Run inference on a leaf image
```bash
python predict.py --image path/to/leaf.jpg
```

Or use it directly in Python:
```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("path/to/leaf.jpg", conf=0.25)
results[0].show()
```

---

## 📊 Data Augmentation

Training used the following augmentations:
- Rotation, horizontal & vertical flips
- Blur, brightness, contrast adjustments
- Mosaic augmentation (built into YOLO training)

---

## 📁 Label Format

Labels follow the standard **YOLO format**:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized between 0 and 1.

---

## 🌍 Real-World Impact

- Enables early disease detection before visible spread
- Can be deployed on mobile devices or drones for field scanning
- Reduces reliance on manual crop inspection
- Directly supports food security and sustainable agriculture

---

## 📌 Status

> ✅ Model trained and tested – inference ready.

---

## 🙏 Dataset Credit

**PlantVillage Dataset**  
Hughes, D.P. & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.*  
Available on [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

*Made by [@sofsof23](https://github.com/sofsof23)*
