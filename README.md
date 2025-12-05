# 🚦 German Traffic Sign Recognition Benchmark (GTSRB) Classifier

![alt text](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGt1aDBsZmRiMG5pNHZ4OWQ2bTJoeWo4MG5kd2tobzI1ZGxkZG11MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0HlKQPTHOGNUPTZm/giphy.gif)

> **A State-of-the-Art Deep Learning classifier achieving 99.6% accuracy using ResNet18 Transfer Learning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed-green.svg)]()

---

## 📌 Project Overview

This project implements a robust computer vision pipeline to classify **43 types of traffic signs** from the famous [GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Unlike traditional approaches that train small CNNs from scratch, this project leverages **Transfer Learning** with a pre-trained **ResNet18** architecture. By adapting the network's input layer for low-resolution ($32 \times 32$) images and fine-tuning the feature extractors, we achieve **superhuman performance**.

### 🚀 Key Results

| Metric                  | Performance | Notes                                            |
| :---------------------- | :---------- | :----------------------------------------------- |
| **Test Accuracy**       | **99.60%**  | Evaluated on the official unseen Test Set        |
| **Validation Accuracy** | **99.97%**  | Evaluated on a 20% hold-out split                |
| **Human Benchmark**     | ~98.84%     | This model outperforms average human recognition |

---

## 🛠️ Methodology

### 1. Data Preprocessing

- **ROI Cropping:** Parsed annotation CSVs to extract the specific Region of Interest (ROI), removing background noise (trees, sky).
- **Augmentation:** Applied rigorous data augmentation to the training set to prevent overfitting:
  - Random Rotations ($\pm 10^\circ$)
  - Affine Translations
  - Color Jitter (Brightness/Contrast)
- **Normalization:** Standardized inputs using ImageNet mean/std statistics.

### 2. Model Architecture: ResNet18 (Modified)

We utilized a standard ResNet18 backbone with two critical modifications for small image handling:

- **The "Eye":** Replaced the initial $7 \times 7$ convolution (stride 2) with a **$3 \times 3$ convolution (stride 1)**. This prevents aggressive downsampling that would destroy details in $32 \times 32$ icons.
- **The "Head":** Replaced the final 1000-class fully connected layer with a custom **43-class classifier**.

### 3. Training Strategy

- **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-4`)
- **Scheduler:** StepLR (Decay learning rate by 0.1 every 5 epochs)
- **Device:** Accelerated training on macOS using MPS (Metal Performance Shaders).

---

## 📂 Project Structure

```bash
gtsrb_classifier/
├── data/                  # Dataset storage (GTSRB)
├── models/                # Saved trained weights (gtsrb_best_model.pth)
├── notebooks/             # Jupyter Notebooks for analysis
│   ├── 01_Data_Exploration.ipynb   # EDA, Class Distribution, ROI Visualization
│   ├── 02_Model_Training.ipynb     # Main Training Loop & Validation
│   └── 03_Results_Analysis.ipynb   # Inference, Confusion Matrix, Error Analysis
├── src/                   # Source code modules
│   ├── dataset.py         # Custom PyTorch Dataset class
│   ├── model.py           # ResNet18 architecture definition
│   ├── train.py           # CLI Training script
│   └── utils.py           # Checkpointing & Metrics helpers
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## 💻 Installation & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/gtsrb-classifier.git](https://github.com/yourusername/gtsrb-classifier.git)
cd gtsrb-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

You can train the model using the command-line interface:

```bash
python src/train.py --epochs 15 --batch_size 64
```

_Or run the notebook `notebooks/02_Model_Training.ipynb` for an interactive experience._

### 4. Evaluate Performance

Run the analysis notebook to generate confusion matrices and test on internet images:

```bash
jupyter notebook notebooks/03_Results_Analysis.ipynb
```

---

## 📊 Visuals

### Confusion Matrix

The diagonal line indicates near-perfect classification across all 43 classes.

![Confusion Matrix](/Users/kaankoc/Projects/gtsrb_classifier/confusion_matrix.png)
_(Replace with actual screenshot of your confusion matrix)_

### Real-World Inference

The model is robust to real-world images when pre-processed correctly (cropped).

|                                  Input                                  |     Prediction     | Confidence |
| :---------------------------------------------------------------------: | :----------------: | :--------: |
| ![Test Image](/Users/kaankoc/Projects/gtsrb_classifier/test_image.jpeg) | **Speed Limit 50** | **99.99%** |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙌 Acknowledgments

- German Traffic Sign Recognition Benchmark (GTSRB) team for the dataset.
- PyTorch and torchvision teams for the pre-trained models.
