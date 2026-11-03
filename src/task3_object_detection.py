
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "unit5"

IMAGE_PATH = DATA_DIR / "photo1.jpeg"
MODEL_PATH = BASE_DIR / "yolov8n.pt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("TASK 3: OBJECT DETECTION")
print("------------------------")

# ==========================================
# 2. LOAD ORIGINAL IMAGE
# ==========================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Check that photo1.jpeg exists inside the data folder."
    )

print("Original image loaded successfully!")

# ==========================================
# 3. HAAR CASCADE FACE DETECTION
# ==========================================

print("\n1. HAAR CASCADE FACE DETECTION")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar Cascade face detector
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade classifier.")

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

haar_image = image.copy()

# Draw bounding boxes around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(
        haar_image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        haar_image,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

haar_count = len(faces)

print("Faces detected by Haar Cascade:", haar_count)

cv2.imwrite(
    str(OUTPUT_DIR / "haar_face_detection.jpg"),
    haar_image
)

# ==========================================
# 4. YOLO OBJECT DETECTION
# ==========================================

print("\n2. YOLO OBJECT DETECTION")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

# Load the pre-trained YOLOv8 nano model
model = YOLO(str(MODEL_PATH))

# Perform object detection
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    verbose=False
)

result = results[0]

# Get annotated image with boxes, labels and confidence
yolo_image = result.plot()

cv2.imwrite(
    str(OUTPUT_DIR / "yolo_detection.jpg"),
    yolo_image
)

# Print all detected objects
yolo_count = 0
person_count = 0

print("\nDetected objects:")

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        label = result.names[class_id]

        x1, y1, x2, y2 = (
            int(value) for value in box.xyxy[0].tolist()
        )

        yolo_count += 1

        if label == "person":
            person_count += 1

        print(
            f"Object: {label}, "
            f"Confidence: {confidence:.2f}, "
            f"Box: ({x1}, {y1}, {x2}, {y2})"
        )

print("\nTotal YOLO objects detected:", yolo_count)
print("YOLO person detections:", person_count)

# ==========================================
# 5. SAVE COMPARISON IMAGE
# ==========================================

# Convert images from BGR to RGB for Matplotlib
haar_rgb = cv2.cvtColor(haar_image, cv2.COLOR_BGR2RGB)
yolo_rgb = cv2.cvtColor(yolo_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(haar_rgb)
plt.title(f"Haar Cascade - Faces: {haar_count}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(yolo_rgb)
plt.title(f"YOLO - Objects: {yolo_count}")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "task3_detection_comparison.png",
    dpi=150
)

plt.close()

# ==========================================
# 6. SAVE TEXT RESULTS
# ==========================================

with open(OUTPUT_DIR / "task3_results.txt", "w") as file:

    file.write("TASK 3: OBJECT DETECTION RESULTS\n")
    file.write("--------------------------------\n")

    file.write(f"Haar Cascade faces detected: {haar_count}\n")
    file.write(f"YOLO objects detected: {yolo_count}\n")
    file.write(f"YOLO person detections: {person_count}\n")

    file.write("\nYOLO Detection Details:\n")

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = result.names[class_id]

            file.write(
                f"{label}: confidence = {confidence:.2f}\n"
            )

print("\nTASK 3 COMPLETED SUCCESSFULLY!")
print("Outputs saved in:", OUTPUT_DIR)
