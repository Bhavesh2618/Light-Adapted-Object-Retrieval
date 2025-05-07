# Voice-Controlled Surgical Tool Detection using YOLO11n

This project demonstrates real-time **surgical tool detection** using the **YOLO11n** model with **voice commands** under varying environmental conditions (like blur, lighting, angles, blood). It supports live detection from a video file and listens continuously for spoken commands such as "forceps", "mayo", "scalpel", etc., to trigger detection of the specified tool.

## 🔧 Features

- 🎤 Voice-controlled object detection (no button presses required)
- 🛠 Detection of surgical tools using YOLO11n
- 💡 Robust under challenging conditions: blur, lighting variations, blood presence, and diverse angles
- 🧠 Multi-threaded continuous listening + real-time detection
- 📼 Real-time detection on looped video stream
- ✅ GPU-accelerated if available

## 🖼️ Example Tools

- Forceps
- Hemostat
- Mayo
- Scalpel
- Syringe
- Stitch Scissors
- Episiotomy Scissors
- Cotton
- Gloves

## 🗂️ Project Structure

```bash
📁 project-root/
│
├── model/                            # Contains trained YOLOv8/YOLO11n model
│   └── best.pt
├── video/                            # Input surgical video(s)
│   └── v5.mp4
├── main.py                           # Main script
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/voice-guided-surgical-detection.git
cd voice-guided-surgical-detection
```

### 2. Install Dependencies

```bash
pip install numpy opencv-python sounddevice scipy SpeechRecognition torch ultralytics
```

### 3. Run the Script

```bash
python main.py
```

The system will begin listening for voice commands. Speak the name of a surgical tool (e.g., "forceps") and it will highlight that object in the video.

## 🎯 Voice Commands

You can say any of the following:
```
"forceps", "hemostat", "mayo", "scalpel", "syringe", 
"stitch scissors", "episiotomy scissors", "cotton", "gloves", "exit"
```

Saying **"exit"** will stop the program.

## 💡 Model Training

The YOLO11n model used here was trained on a **custom dataset of 6000+ surgical tool images** captured under edge conditions:
- Low light and overexposure
- Motion blur
- Blood-stained scenes
- Varied angles

The trained model achieved **92.7% accuracy** under these conditions.

## 🖥️ Hardware

- Compatible with both CPU and GPU
- Real-time performance is better on CUDA-compatible GPU

## 📜 License

This project is open-source and available under the MIT License.

---

### 👨‍⚕️ Made with ❤️ for surgical AI research.
