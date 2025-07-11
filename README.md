# Triplet Dataset Preparation for Knowledge Distillation

This script generates a **robust triplet dataset** for image enhancement via knowledge distillation.

## 👨‍🏫 Triplet Structure
- **HR (Ground Truth)**
- **Teacher Input** (Upscaled blurry/noisy image)
- **Student Input** (Heavily degraded image)

## 🔧 Configuration
Edit the following in `Data_Preparation.py`:
```python
SOURCE_DIR = "path_to_raw_hr_images"
TARGET_DIR = "path_to_save_triplets"

Demovideolink: "https://drive.google.com/file/d/1DtYNhPgV0uzrxvZj8Otxii59m0jB-zRC/view?usp=drive_link"

Student model weights:"https://drive.google.com/file/d/1QcGf0L901PBnFqMtK4Vd1O-_PekXSWKf/view?usp=drive_link"

Teacher model weights: "https://drive.google.com/file/d/1CysQxuX-5vE33zH9BTG-a9dFszPCwrIf/view?usp=drive_link"


#### 💾 Dataset

The custom dataset created for this project, containing the HR, student_lr, and teacher_lr image triplets, is publicly available on Kaggle.

- **Kaggle Dataset Link:** [Image Sharpening - Custom Video Frames Dataset]("https://www.kaggle.com/datasets/chetanreddyc/ultralightweight-model-training-dataset-v8")
### Refer to the report for DataSource Link and DemoRecording and clear details.

# Real-Time Image Sharpening using Knowledge Distillation

This repository contains the complete project for developing a lightweight, real-time AI model to enhance image sharpness in video conferencing scenarios, optimized with Intel's OpenVINO™ Toolkit.

![Best Case Result](results/best_case_comparison.png)
*An example of the model's performance, showing the degraded input, the enhanced output, and the ground truth.*

---

## 📝 Project Objective

The goal of this project was to develop an efficient deep learning model capable of restoring clarity and sharpness to low-quality video frames caused by low bandwidth or poor network conditions. The final model was required to be fast enough for real-time applications (30-60 FPS) while maintaining high visual quality.

## 🚀 Key Features

- **Knowledge Distillation:** A large "Teacher" U-Net model was used to train a small, lightweight "Student" RRDBNet model.
- **Robust Custom Dataset:** A specialized dataset was created to simulate real-world video artifacts, including compression, noise, and mixed-resolution blur.
- **High-Performance Inference:** The final model is benchmarked on both GPU (with PyTorch) and CPU (with OpenVINO) to demonstrate its efficiency.
- **Real-Time Capable:** The optimized model achieves real-time performance on standard CPU hardware, making it suitable for broad deployment.

---

## 🏛️ Model Architectures

- **Teacher (U-Net):** A wide U-Net architecture was chosen for its strength in understanding image context and producing a high-quality, structurally accurate target for the student to learn from.
- **Student (RRDBNet):** A lightweight RRDBNet was chosen for its exceptional ability to generate realistic, high-frequency textures and its computationally efficient design, which performs heavy processing at low resolution.

---

## 📊 Final Performance Results

The final Student model was benchmarked on a diverse validation set, yielding the following results:

| Hardware | Framework | Pure Model FPS | Quality (SSIM) |
| :--- | :--- | :---: | :---: |
| **NVIDIA T4 GPU** | PyTorch | ~58.5 FPS | 0.878 |
| **Intel CPU** | PyTorch (Baseline) | ~4.2 FPS | 0.894 |
| **Intel CPU** | **OpenVINO™ (Optimized)** |   | 0.894 |

Conclusion: The OpenVINO™ optimization provided a **~20x performance increase** on the CPU, successfully elevating the model to real-time capability on standard hardware.

---

## 🛠️ How to Use This Repository

The project is organized into several notebooks, designed to be run in a cloud environment like Google Colab.

### 1. Data Preparation
- The script in `1_Data_Preparation/` shows the pipeline used to create the training dataset from source images.

### 2. Model Training
- The notebook in `StudentModel_pytorch/` contains the complete code for training the Student model using knowledge distillation from the Teacher.

### 3. Benchmarking
- The notebooks in `3_Benchmarking/` allow you to reproduce our performance tests.
  - `Benchmarking\Student_BenchMark.ipynb` for GPU testing.
  - `Benchmark_OpenVINO_CPU.ipynb` for optimized CPU testing.

### 4. Live Demo
- The `StudentModel__Pytorch\Benchmarking\KD_livefeedtest.ipynb/` notebook provides a live, side-by-side webcam demo to visualize the model's real-time enhancement capabilities.
which showed the results of approximately 130 Model Fps and 5-6 system FPS.

### For visual outputs refer to the "Results folder" 


---



