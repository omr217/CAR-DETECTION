# 🚀 Real-Time Object Detection with TensorRT

This project performs real-time object detection using a **TensorRT-optimized deep learning model**. It loads a precompiled `.trt` engine, runs inference on a video source, and displays detections with class labels and FPS.

---

## 📦 Features

- 🔧 Runs high-performance inference using NVIDIA **TensorRT**
- 🖼️ Supports any YOLO-style model with bounding box outputs
- 🎥 Processes video input and overlays bounding boxes with labels
- 🚀 Uses **GPU acceleration** via `pycuda`
- 📊 Displays real-time FPS on the output window

---

## 🛠 Requirements

Ensure you have the following installed:

- Python 3.7+
- OpenCV
- NumPy
- TensorRT
- PyCUDA

### Installation (Ubuntu/Linux)

```bash
pip install numpy opencv-python pycuda
# TensorRT must be installed separately from NVIDIA
```

#### 🗃 Project Structure
.
├── model.trt          # Precompiled TensorRT engine file
├── cars.mp4           # Input video for inference
├── detect_trt.py      # Main Python script
├── README.md

##### 🧠 Model Assumptions
This project assumes that your TensorRT engine outputs detections in the format:
```
[x1, y1, x2, y2, confidence, class_id]
```

###### ⚠️ Notes
This script assumes you have a valid .trt engine created using TensorRT APIs.

The engine input and output shapes must match the preprocessing and postprocessing logic.

The model should be optimized using FP16 or INT8 for best performance on Jetson/NVIDIA GPUs.

