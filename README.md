# Computer Vision – Face Detection Project

## 1. Project Title

**Face Detection Using Python and OpenCV**

## 2. Project Description

This project is developed as part of the CCA 3 Practical Coding Assignment for Computer Vision.

The project demonstrates fundamental computer vision techniques using Python and OpenCV. It focuses on detecting human faces from images and self-recorded videos using computer vision algorithms.

The system processes input images and video frames, identifies human faces, and highlights the detected faces using rectangular bounding boxes.

This project provides practical experience in image processing, video processing, and real-time computer vision applications.

## 3. Objectives

The main objectives of this project are:

- To understand the fundamentals of computer vision.
- To implement face detection using Python and OpenCV.
- To read and process input images.
- To capture and process video frames.
- To detect human faces using a pre-trained Haar Cascade classifier.
- To highlight detected faces using bounding boxes.
- To understand practical applications of computer vision techniques.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV | Image and video processing |
| NumPy | Numerical operations |
| Matplotlib | Image visualization |
| Haar Cascade | Face detection |
| VS Code | Development environment |
| Git & GitHub | Version control and project submission |

## 5. Project Folder Structure

```text
CV_FaceDetectionProject/
│
├── src/
│   └── main.py
│
├── images/
│   ├── input/
│   └── output/
│
├── videos/
│   ├── input/
│   └── output/
│
├── screenshots/
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

**Folder Description:**

- `src/` – Contains the Python source code.
- `images/input/` – Contains input photographs.
- `images/output/` – Stores processed images with detected faces.
- `videos/input/` – Contains self-recorded input videos.
- `videos/output/` – Stores processed videos with detected faces.
- `screenshots/` – Contains screenshots of project execution and outputs.
- `requirements.txt` – Contains the required Python libraries.
- `.gitignore` – Specifies files and folders excluded from GitHub.
- `README.md` – Contains project documentation.

## 6. Installation Instructions

### Prerequisites

- Python 3.x
- Visual Studio Code
- Git
- Internet connection for installing required libraries

### Step 1: Clone the Repository

```bash
git clone https://github.com/sunilsonawane29/CV_FaceDetectionProject.git
```

### Step 2: Navigate to the Project Folder

```bash
cd CV_FaceDetectionProject
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Required Libraries

```bash
pip install -r requirements.txt
```

If required, install the libraries manually:

```bash
pip install opencv-python numpy matplotlib scikit-image
```

## 7. How to Run the Four Tasks

Open the project folder in VS Code and activate the virtual environment before executing the Python program.

### Task 1: Face Detection from Image

**Objective:** Detect human faces from an input photograph.

**Steps:**
1. Place the input photograph inside `images/input/`.
2. Run the Python program.
3. The program loads the photograph.
4. OpenCV detects faces using the Haar Cascade classifier.
5. Bounding boxes are drawn around detected faces.
6. The processed image is saved inside `images/output/`.

### Task 2: Face Detection from Video

**Objective:** Detect human faces from a self-recorded video.

**Steps:**
1. Place the self-recorded video inside `videos/input/`.
2. Run the video processing program.
3. The program reads the video frame by frame.
4. Each frame is converted into grayscale.
5. Faces are detected using the Haar Cascade classifier.
6. Rectangular bounding boxes are drawn around detected faces.
7. The processed video is saved inside `videos/output/`.

### Task 3: Real-Time Face Detection

**Objective:** Detect human faces using a live webcam feed.

**Steps:**
1. Connect or enable the laptop webcam.
2. Execute the real-time face detection program.
3. The webcam captures live video frames.
4. OpenCV processes each frame.
5. Detected faces are highlighted with rectangular boxes.
6. Press `Q` to stop the webcam window, if supported by the program.

### Task 4: Face Detection Output and Visualization

**Objective:** Display and analyze the processed face detection results.

**Steps:**
1. Open the generated output images or videos.
2. View the detected faces and their bounding boxes.
3. Use Matplotlib to display the processed image, if implemented.
4. Save screenshots of the successful execution.
5. Store the screenshots inside the `screenshots/` folder.

*Note: Run the corresponding task-specific Python files or functions if your implementation separates the four tasks. The steps above describe the intended workflow.*

## 8. Sample Output Screenshots

The following screenshots demonstrate the expected project outputs.

### Screenshot 1: Input Image

Add your original input photograph here.

### Screenshot 2: Image Face Detection

Add your processed image showing detected faces and bounding boxes here.

### Screenshot 3: Video Face Detection

Add a screenshot showing face detection from your self-recorded video here.

### Screenshot 4: Real-Time Face Detection

Add a screenshot of the webcam face detection window here.

### Screenshot 5: Project Execution

Add a screenshot of successful program execution in VS Code or the terminal here.

## 9. Expected Results

The project demonstrates the ability to:

- Load and process images using OpenCV.
- Read and process video files.
- Detect human faces using a Haar Cascade classifier.
- Draw rectangular bounding boxes around detected faces.
- Visualize computer vision outputs.
- Apply basic computer vision concepts to practical applications.

## 10. Applications

- Attendance management systems.
- Video surveillance systems.
- Security applications.
- Human-computer interaction.
- Face-based monitoring systems.
- Computer vision applications.

## 11. Author Information

**Name:** Sunil Sonawane  
**Course:** Master of Computer Applications (MCA)  
**College:** MIT World Peace University, Pune  
**Assignment:** CCA 3 – Computer Vision Practical Coding Assignment  
**Programming Language:** Python

**GitHub:** https://github.com/sunilsonawane29

---

## 12. Conclusion

This project helped in understanding the practical implementation of face detection using Python and OpenCV.

It provided hands-on experience with image processing, video processing, Haar Cascade classifiers, and computer vision techniques.

The project demonstrates the basic working principles of face detection and its applications in real-world scenarios.

## 13. Project Explanation Video

The complete project demonstration and explanation
video is available at the link below:

[Watch CCA 3 Project Explanation Video](https://drive.google.com/file/d/1zvLzZXZ_iRv4vepXY0QJ8xaFuBBy1P-c/view?usp=sharing)