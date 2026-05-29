# 🩺 DeepDiabetic — Diabetic Eye Disease Identification System

> A multi-class deep learning framework for automated detection of diabetic eye diseases from fundus images, achieving up to **98.76% accuracy** using EfficientNetB0.

[![IEEE](https://img.shields.io/badge/Published-IEEE%20Access%202024-blue)](https://ieeexplore.ieee.org/document/10401038/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Disease Classes](#disease-classes)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Authors](#authors)

---

## Overview

**DeepDiabetic** is a deep learning-based identification system that classifies four major diabetic eye diseases from retinal fundus images. The system addresses the limitations of traditional manual screening methods — which are slow, subjective, and error-prone — by automating early detection using state-of-the-art neural network architectures.

The project benchmarks five supervised deep learning models across multiple evaluation metrics and data augmentation strategies to identify the best-performing classifier.

---

## Disease Classes

The system classifies the following four conditions:

| Class | Description |
|-------|-------------|
| **Diabetic Retinopathy (DR)** | Damage to retinal blood vessels caused by prolonged high blood sugar |
| **Diabetic Macular Edema (DME)** | Swelling in the macula caused by fluid leakage from retinal blood vessels |
| **Glaucoma** | Damage to the optic nerve, often associated with increased eye pressure |
| **Cataract** | Clouding of the eye's natural lens, leading to blurry or dim vision |

---

## Models

Five deep learning architectures were developed and evaluated:

### CNN-Based (Transfer Learning)

| Model | Description |
|-------|-------------|
| **EfficientNetB0** | Lightweight, compound-scaled CNN — best overall performer |
| **VGG16** | Classic 16-layer deep convolutional network |
| **ResNet152V2** | 152-layer residual network with skip connections |

### RNN-Hybrid Models

| Model | Description |
|-------|-------------|
| **GRU + ResNet152V2** | Gated Recurrent Unit combined with ResNet feature extraction |
| **Bi-GRU + ResNet152V2** | Bidirectional GRU for richer sequential feature representation |

---

## Dataset

- **Total Images:** 1,228 retinal fundus images
- **Classes:** 4 (DR, DME, Glaucoma, Cataract)
- **Augmentation Strategies:** Three variants evaluated per model:
  - Non-augmented images
  - Offline augmented images
  - Online augmented images

---

## Results

Performance was measured on accuracy, precision, recall, loss, and AUC.

| Model | Accuracy |
|-------|----------|
| **EfficientNetB0** | **98.76%** ✅ Best |
| GRU + ResNet152V2 | 98.38% |
| Bi-GRU + ResNet152V2 | — |
| VGG16 | — |
| ResNet152V2 | Lowest |

> EfficientNetB0 achieved the highest classification performance across all evaluated metrics.

---

## Project Structure

```
DeepDiabetic/
│
├── data/
│   ├── raw/                    # Original fundus images
│   ├── augmented_offline/      # Offline-augmented images
│   └── augmented_online/       # Preprocessing for online augmentation
│
├── models/
│   ├── efficientnetb0.py       # EfficientNetB0 model definition
│   ├── vgg16.py                # VGG16 model definition
│   ├── resnet152v2.py          # ResNet152V2 model definition
│   ├── gru_resnet.py           # GRU + ResNet hybrid model
│   └── bigru_resnet.py         # Bi-GRU + ResNet hybrid model
│
├── notebooks/
│   ├── EfficientNetB0.ipynb    # Training & evaluation notebook
│   ├── VGG16.ipynb
│   ├── ResNet152V2.ipynb
│   ├── GRU_ResNet.ipynb
│   └── BiGRU_ResNet.ipynb
│
├── utils/
│   ├── preprocessing.py        # Image preprocessing utilities
│   ├── augmentation.py         # Data augmentation helpers
│   └── evaluation.py           # Metrics and plotting functions
│
├── results/
│   ├── metrics/                # Accuracy, precision, recall, AUC tables
│   └── figures/                # Training curves and confusion matrices
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/DeepDiabetic.git
cd DeepDiabetic

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

```
tensorflow>=2.10
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
```

---

## Usage

### Training a Model

```python
from models.efficientnetb0 import build_model
from utils.preprocessing import load_dataset

# Load and preprocess data
X_train, X_val, y_train, y_val = load_dataset("data/augmented_offline/")

# Build and train model
model = build_model(num_classes=4, input_shape=(224, 224, 3))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
```

### Running Inference

```python
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image

model = load_model("results/efficientnetb0_best.h5")
image = preprocess_image("path/to/fundus_image.jpg")

prediction = model.predict(image)
classes = ["DR", "DME", "Glaucoma", "Cataract"]
print(f"Predicted: {classes[prediction.argmax()]}")
```

### Jupyter Notebooks

Open any notebook in the `notebooks/` folder to explore model training, evaluation curves, and confusion matrices interactively:

```bash
jupyter notebook notebooks/EfficientNetB0.ipynb
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{albelaihi2024deepdiabetic,
  title     = {DeepDiabetic: An Identification System of Diabetic Eye Diseases Using Deep Neural Networks},
  author    = {Albelaihi, A. and Ibrahim, D. M.},
  journal   = {IEEE Access},
  year      = {2024},
  doi       = {10.1109/ACCESS.2024.XXXXXXX},
  url       = {https://ieeexplore.ieee.org/document/10401038/}
}
```

---

## Authors

- **A. Albelaihi** — Research & Model Development
- **D. M. Ibrahim** — Research & Model Development

Published in **IEEE Access**, February 2024.

---

## License

This project is released for academic and research purposes. Please refer to the [LICENSE](LICENSE) file for full terms.

---

*For questions, issues, or contributions, please open a GitHub issue or submit a pull request.*
