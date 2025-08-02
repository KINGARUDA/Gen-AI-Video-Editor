# 🎬 Gen-AI Video Editor

**Authors:** Lucky Verma, Manvendra Singh  
**Date:** June 28, 2025

This project enhances video quality and artistic style using deep learning. It integrates models for **super-resolution**, **frame interpolation**, **neural style transfer (with/without temporal consistency)**, and **lip-synced video generation**, all within a simple **Gradio UI**.


🔗 [GitHub Repo](https://github.com/KINGARUDA/Gen-AI-Video-Editor)  
📁 [Download Sample Outputs / Weights](https://drive.google.com/file/d/1pwUDQyk5ZH1BK4EL3q8ECmBcJw6tY6JL/view?usp=sharing) 

---

## 🌟 Demo

<!-- Replace below with actual demo video/image -->
![Sample Output](docs/sample_output.gif)
> _Above: Stylized and enhanced video output_

---

## 📌 Features

- **Super-Resolution** with SRGAN and SOFVSR
- **Frame Interpolation** with RIFE
- **Artistic Stylization** with ReCoNet and AnimeGANv2
- **Lip-Sync Generation** using Lip2Wav
- **User Interface** built with Gradio
- Modular, pipeline-friendly design

---

## 🧠 Methodology

### 🔹 1. Video Super-Resolution
- **SRGAN**: Used for single-frame enhancement.
- **SOFVSR**: A spatio-temporal model to preserve consistency across video frames.

### 🔹 2. Frame Interpolation
- **RIFE**: Real-time optical flow-based frame interpolator for smooth playback and slow-motion effects.

### 🔹 3. Video Style Transfer
- **ReCoNet**: Maintains temporal consistency in stylized videos.
- **AnimeGANv2**: Fast anime-style rendering (frame-by-frame).

### 🔹 4. Lip-Synced Video Generation
- **Lip2Wav**: Generates synchronized lip movements from speech audio and a reference video.

---

## ⚙️ System Design

- **Language**: Python
- **Libraries**: PyTorch, OpenCV, ffmpeg
- **Interface**: Gradio for interactive UI
- **Video Processing**: Frame-by-frame, then recompiled to video

---

## 🚀 Results

| Module            | Outcome Summary                                                  |
|-------------------|------------------------------------------------------------------|
| **SOFVSR**        | Enhanced video sharpness with temporal stability                 |
| **RIFE**          | Smooth motion generation with intermediate frames                |
| **ReCoNet**       | Temporally consistent stylized outputs                           |
| **AnimeGANv2**    | Visually compelling stylization (cartoon/comic feel)             |
| **Lip2Wav**       | Accurate mouth movement synced to input speech                   |

---

## 🧪 How to Run

```bash
Install the zip file from google drive and set it as working directory


# Install requirements
pip install -r requirements.txt

# Run the Gradio UI
python app.py


