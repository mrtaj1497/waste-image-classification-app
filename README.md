# ♻️ Automated Waste Classification Using Deep Learning for Smart Recycling

An AI-powered waste detection and classification system built using **YOLO11, Roboflow, Python, and Gradio**. The application detects and classifies different types of waste from uploaded images and videos using object detection.

The system is designed to support smarter waste segregation and improve automated recycling processes.

## 🚀 Project Overview

Traditional waste sorting is time-consuming and often requires manual effort. This project explores an automated approach using deep learning to identify and locate different waste objects in real-world environments.

The application uses a custom-trained YOLO11 model hosted on Roboflow to detect multiple waste categories and display predictions with bounding boxes and confidence scores.

## ✨ Key Features

- **Image Detection:** Upload an image to detect and classify waste objects.
- **Video Detection:** Upload a video to perform frame-based waste detection.
- **Multi-Class Detection:** Recognizes eight different waste categories.
- **Bounding Box Visualization:** Displays detected objects with class labels and confidence scores.
- **Detection Summary:** Provides a summary of detected waste categories.
- **Interactive Web Interface:** Built with Gradio.
- **Roboflow Integration:** Uses hosted model inference through the Roboflow API.

## 🗂️ Waste Categories

The model supports eight waste categories:

| Class | Description |
|---|---|
| E_waste | Electronic waste |
| Organic | Organic waste |
| Paper | Paper-based waste |
| Textile | Fabric and textile waste |
| glass | Glass waste |
| metal | Metal waste |
| mixed | Mixed waste |
| plastics | Plastic waste |

## 🛠️ Technologies Used

- Python
- YOLO11
- Roboflow
- Roboflow Inference SDK
- Gradio
- OpenCV
- NumPy
- Pillow
- Python-dotenv

## 🏗️ System Architecture

1. User uploads an image or video.
2. The Gradio application receives the input.
3. The application sends image data to the hosted Roboflow inference service.
4. The YOLO11 model processes the input and generates object detection predictions.
5. Predictions are received by the application.
6. OpenCV draws bounding boxes, class labels, and confidence scores.
7. The final detection results are displayed through the web interface.

## 📁 Project Structure

```text
waste-app/
│
├── app.py                 # Main Gradio application
├── .env                   # Roboflow API key (not committed)
├── .gitignore             # Files excluded from Git
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

### 2. Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY
```

### 3. Create a Virtual Environment

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
python -m pip install -r requirements.txt
```

### 5. Configure the Roboflow API Key

Create a `.env` file in the root project directory:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key
```

**Important:** Obtain your API key from your Roboflow account. Never upload your API key or `.env` file to GitHub.

### 6. Run the Application

```bash
python app.py
```

The application will start locally and open in your default browser.

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
gradio
inference-sdk
opencv-python
numpy
Pillow
python-dotenv
```

## 🖥️ Application Modules

### 1. Image Detection

Users can upload an image through the web interface. The system processes the image and displays detected waste objects with their corresponding class names and confidence scores.

### 2. Video Detection

Users can upload a video file. The application processes selected video frames, performs object detection, and generates an annotated video showing the detected waste objects.

## 📊 Model Information

| Parameter | Details |
|---|---|
| Model Architecture | YOLO11 |
| Model Variant | YOLO11 Nano |
| Model Platform | Roboflow |
| Model ID | waste-image-classification-298ab/15 |
| Number of Classes | 8 |
| Input Types | Images and Videos |
| Detection Type | Multi-Class Object Detection |

## 📈 Evaluation Metrics

The model can be evaluated using the following object detection metrics:

- Precision
- Recall
- mAP@50
- mAP@50:95
- F1-score

These metrics help assess the model's ability to detect and classify waste objects.

## 🔒 Security

- Store API credentials in a local `.env` file.
- Never commit API keys or private credentials.
- Add `.env` and `.venv/` to `.gitignore`.

Example `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
.DS_Store
```

## ⚠️ Limitations

- Detection performance depends on image quality and object visibility.
- Heavily overlapping or partially hidden waste objects can be difficult to detect.
- Video processing speed depends on frame sampling and hardware resources.
- The application performs detection and classification; it does not physically sort waste.

## 🎯 Future Improvements

- Expand the dataset with more diverse real-world waste images.
- Improve detection of small and heavily overlapping waste objects.
- Optimize inference speed for faster video processing.
- Explore deployment on edge devices for practical waste management applications.

## 👨‍💻 Project Information

**Project Title:** Automated Waste Classification Using Deep Learning for Smart Recycling

**Department:** Computer Science and Engineering (CSE)

**University:** Daffodil International University

**Project Focus:** Deep Learning, Computer Vision, YOLO11, Object Detection, and Smart Waste Management.

---

## ⭐ Acknowledgement

We acknowledge the contributions of the open-source community, Ultralytics, and Roboflow for providing the tools and technologies that supported this project.

**If you find this project useful, consider giving the repository a star!** ♻️
