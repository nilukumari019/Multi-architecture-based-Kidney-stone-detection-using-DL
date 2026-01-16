🩺 Kidney Stone Detection from CT Images using Deep Learning
📌 Overview

This project presents a multi-architecture deep learning framework for automated kidney stone detection from axial CT images. The system combines multiple pre-trained convolutional neural networks (CNNs) to improve classification performance while ensuring robust generalization and leakage-free evaluation.

The pipeline is designed specifically for medical imaging tasks, incorporating contrast enhancement, balanced augmentation, and rigorous evaluation metrics.

🎯 Key Objectives

Detect kidney stones from CT images using deep learning

Improve generalization using medical-specific preprocessing

Prevent data leakage between training and testing

Evaluate performance using clinically relevant metrics

🧠 Model Architecture

The proposed framework uses feature fusion from multiple CNN backbones:

VGG19 – captures fine-grained spatial features

ResNet50 – learns deep residual representations

EfficientNet-B0 – balances accuracy and efficiency

The extracted features are:

Spatially aligned

Channel-reduced

Fused using convolution layers

Refined with a global attention mechanism

Classified using a fully connected network

🧪 Dataset

Dataset: Axial CT Imaging Dataset for Kidney Stone Detection

Classes:

Non-Stone

Stone

Dataset Handling

Training & Validation: Augmented dataset (5× balanced augmentation)

Testing: Strictly original, unseen images

Splitting Strategy: Stratified splitting to avoid class imbalance and data leakage

🛠️ Preprocessing Pipeline

Medical-specific preprocessing is applied to each CT image:

CLAHE (Contrast Limited Adaptive Histogram Equalization)
Enhances contrast while preserving anatomical details.

Median Filtering
Reduces noise without blurring edges.

🔄 Data Augmentation

To improve generalization and handle limited medical data:

Offline Augmentation (5×)

Horizontal flipping

Rotation

Random resized cropping

Brightness and contrast adjustments

Online Augmentation (During Training)

Random flips

Rotations

Normalization using ImageNet statistics

⚙️ Training Strategy

Transfer Learning using ImageNet-pretrained weights

Class-weighted loss to address imbalance

Adam optimizer with learning-rate scheduling

Early stopping based on validation performance

Batch size: 8

Image size: 224 × 224

📊 Evaluation Metrics

The model is evaluated using:

Accuracy

ROC–AUC

Confusion Matrix

Precision, Recall, F1-score

All evaluations are performed on a strictly unseen original test set.

📈 Results

Achieved high classification performance

ROC–AUC ≈ 0.99

Stable training and validation curves

Minimal overfitting due to balanced augmentation and regularization

🧰 Tech Stack

Programming Language: Python

Deep Learning: PyTorch, Torchvision

Image Processing: OpenCV, PIL

Data Handling: NumPy, scikit-learn

Visualization: Matplotlib, Seaborn

📂 Project Structure
├── dataset/
│   ├── Non-Stone/
│   └── Stone/
├── AUGMENTED_DATASET/
│   ├── Non-Stone/
│   └── Stone/
├── training_script.py
├── plots/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_metrics_plot.png
└── README.md

▶️ How to Run

Clone the repository:

git clone "https://github.com/nilukumari019/Multi-architecture-based-Kidney-stone-detection-using-DL"


Install dependencies:

pip install torch torchvision numpy opencv-python scikit-learn matplotlib seaborn tqdm


Update dataset path in the script:

DATASET_PATH ="https://www.kaggle.com/datasets/orvile/axial-ct-imaging-dataset-kidney-stone-detection"


Run the training script:

python training_script.py

🚀 Future Improvements

Patient-level data splitting

Model ablation studies

Lightweight deployment models

Clinical validation on multi-center datasets
⭐ If you find this project useful, consider giving it a star!
