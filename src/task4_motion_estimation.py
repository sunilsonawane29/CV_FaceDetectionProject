
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "unit5" / "task4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

video_files = sorted(
    file for file in DATA_DIR.iterdir()
    if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
)

if not video_files:
    raise FileNotFoundError(
        "No video found in the data folder. "
        "Copy your self-recorded video into data."
    )

VIDEO_PATH = video_files[0]

print("TASK 4: MOTION ESTIMATION")
print("-------------------------")
print("Input video:", VIDEO_PATH.name)

# ==========================================
# 2. OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open the input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 25.0

print("Video FPS:", fps)
print("Video dimensions:", width, "x", height)
print("Total frames:", total_frames)

# ==========================================
# 3. CREATE OUTPUT VIDEO WRITERS
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

frame_diff_writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "frame_differencing.mp4"),
    fourcc,
    fps,
    (width, height)
)

gmm_writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "gmm_background_subtraction.mp4"),
    fourcc,
    fps,
    (width, height)
)

optical_writer = cv2.VideoWriter(
    str(OUTPUT_DIR / "lucas_kanade_optical_flow.mp4"),
    fourcc,
    fps,
    (width, height)
)

if not all([
    frame_diff_writer.isOpened(),
    gmm_writer.isOpened(),
    optical_writer.isOpened()
]):
    raise RuntimeError("Could not create output videos.")

# ==========================================
# 4. INITIALIZE ALGORITHMS
# ==========================================

# GMM background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=True
)

# Variables for Lucas-Kanade optical flow
previous_gray = None
previous_points = None

feature_params = {
    "maxCorners": 200,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}

lk_params = {
    "winSize": (15, 15),
    "maxLevel": 2,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
}

# Save representative frames
saved_frame_diff = False
saved_gmm = False
saved_optical = False

frame_number = 0
processed_frames = 0

# ==========================================
# 5. PROCESS VIDEO FRAME BY FRAME
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------------
    # TECHNIQUE 1: FRAME DIFFERENCING
    # --------------------------------------

    if previous_gray is not None:

        difference = cv2.absdiff(previous_gray, gray)

        # Threshold to identify moving regions
        _, motion_mask = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )

        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        frame_diff_output = cv2.cvtColor(
            motion_mask,
            cv2.COLOR_GRAY2BGR
        )

        frame_diff_writer.write(frame_diff_output)

        if not saved_frame_diff:
            cv2.imwrite(
                str(OUTPUT_DIR / "frame_difference_result.jpg"),
                frame_diff_output
            )
            saved_frame_diff = True

    else:
        motion_mask = np.zeros_like(gray)

    # --------------------------------------
    # TECHNIQUE 2: GMM BACKGROUND SUBTRACTION
    # --------------------------------------

    foreground_mask = background_subtractor.apply(frame)

    # Remove shadows and keep foreground pixels
    _, foreground_mask = cv2.threshold(
        foreground_mask,
        200,
        255,
        cv2.THRESH_BINARY
    )

    gmm_output = cv2.cvtColor(
        foreground_mask,
        cv2.COLOR_GRAY2BGR
    )

    gmm_writer.write(gmm_output)

    if not saved_gmm:
        cv2.imwrite(
            str(OUTPUT_DIR / "gmm_foreground_mask.jpg"),
            gmm_output
        )
        saved_gmm = True

    # --------------------------------------
    # TECHNIQUE 3: LUCAS-KANADE OPTICAL FLOW
    # --------------------------------------

    optical_output = frame.copy()

    if previous_gray is None:
        previous_points = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            **feature_params
        )

    else:

        if previous_points is not None and len(previous_points) > 0:

            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                previous_points,
                None,
                **lk_params
            )

            if next_points is not None and status is not None:

                good_new = next_points[status.flatten() == 1]
                good_old = previous_points[status.flatten() == 1]

                for new, old in zip(good_new, good_old):

                    x_new, y_new = new.ravel()
                    x_old, y_old = old.ravel()

                    cv2.line(
                        optical_output,
                        (int(x_old), int(y_old)),
                        (int(x_new), int(y_new)),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        optical_output,
                        (int(x_new), int(y_new)),
                        4,
                        (0, 0, 255),
                        -1
                    )

                previous_points = good_new.reshape(-1, 1, 2)

            else:
                previous_points = None

        # Find new tracking points if necessary
        if previous_points is None or len(previous_points) < 10:

            previous_points = cv2.goodFeaturesToTrack(
                gray,
                mask=None,
                **feature_params
            )

    optical_writer.write(optical_output)

    if not saved_optical:
        cv2.imwrite(
            str(OUTPUT_DIR / "optical_flow_result.jpg"),
            optical_output
        )
        saved_optical = True

    previous_gray = gray.copy()

    processed_frames += 1

    if frame_number % 50 == 0:
        print("Processed frames:", frame_number)

# ==========================================
# 6. RELEASE VIDEO RESOURCES
# ==========================================

cap.release()

frame_diff_writer.release()
gmm_writer.release()
optical_writer.release()

# ==========================================
# 7. CREATE COMPARISON IMAGE
# ==========================================

result_images = [
    OUTPUT_DIR / "frame_difference_result.jpg",
    OUTPUT_DIR / "gmm_foreground_mask.jpg",
    OUTPUT_DIR / "optical_flow_result.jpg"
]

titles = [
    "Frame Differencing",
    "GMM Background Subtraction",
    "Lucas-Kanade Optical Flow"
]

if all(path.exists() for path in result_images):

    plt.figure(figsize=(15, 6))

    for index, (path, title) in enumerate(
        zip(result_images, titles)
    ):

        result_image = cv2.imread(str(path))
        result_image = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )

        plt.subplot(1, 3, index + 1)
        plt.imshow(result_image)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "task4_comparison.png",
        dpi=150
    )

    plt.close()

# ==========================================
# 8. SAVE TEXT RESULTS
# ==========================================

with open(OUTPUT_DIR / "task4_results.txt", "w") as file:

    file.write("TASK 4: MOTION ESTIMATION RESULTS\n")
    file.write("---------------------------------\n")
    file.write(f"Input video: {VIDEO_PATH.name}\n")
    file.write(f"FPS: {fps}\n")
    file.write(f"Video dimensions: {width} x {height}\n")
    file.write(f"Total frames in video: {total_frames}\n")
    file.write(f"Processed frames: {processed_frames}\n")

    file.write("\nAlgorithms implemented:\n")
    file.write("1. Frame Differencing\n")
    file.write("2. GMM Background Subtraction using MOG2\n")
    file.write("3. Lucas-Kanade Optical Flow\n")

print("\nTASK 4 COMPLETED SUCCESSFULLY!")
print("Processed frames:", processed_frames)
print("Outputs saved in:", OUTPUT_DIR)
