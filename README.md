\# AI-Powered System for Finding Crop Diseases



\#Summary
\-Built a deep learning-based image classification system using transfer learning (EfficientNet) to detect crop diseases from leaf images. The model is deployed via a Gradio interface for real-time inference, enabling users to upload images and receive predictions instantly.


\#Features

\-Upload a picture of a leaf

\-Use deep learning to guess what kind of disease will affect crops

\-Gradio-based interactive web interface

\-Quick CPU inference


\#Model Information

\-Model: EfficientNet(Learning by Transfer)

\-Framework: PyTorch

\-The dataset is called PlantVillage

\-Size of Input: 128x128

\-Model performance varied between 45–70% depending on class imbalance and training configuration. Focus was placed on understanding transfer learning behavior and deployment rather than aggressive hyperparameter tuning



\#Tech Stack

\-Python

\-PyTorch

\-torchvision

\-timm

\-Gradio

\-PIL

\#Model Training Details

\-Used transfer learning with EfficientNet pretrained on ImageNet

\-Fine-tuned final layers for multi-class classification

\-Applied data preprocessing (resizing, normalization)

\-Used cross-entropy loss and Adam optimizer

\#Computer Vision Concepts Used

\-Image preprocessing and normalization

\-Transfer learning in CNNs

\-Feature extraction using pretrained models

\-Real-time inference pipeline

\#How the project is setup

crop-disease-project/

│

├── src/

│ ├── data\_loader.py

│ ├── model.py

│ ├── train.py

│ ├── predict.py

│ └── app.py

│

├── outputs/

│ └── model.pth

│

├── data/ (not included in repo)

├── README.md

└── requirements.txt



\#How to run

1. Clone Repository

git clone <your-repo-link>

cd crop-disease-project



2\. Create Virtual Environment

python -m venv crop\_env

crop\_env\\Scripts\\activate



3\. Install Dependencies

pip install -r requirements.txt



4\. Run App

python src/app.py

#App Screenshot
![App Screenshot](screenshot.png)
