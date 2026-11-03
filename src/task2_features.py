
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.feature import hog


# ==========================================
# 1. SET PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "unit4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD TWO PHOTOGRAPHS
# ==========================================

extensions = ["*.jpg", "*.jpeg", "*.png"]

image_files = []

for ext in extensions:
    image_files.extend(DATA_DIR.glob(ext))

image_files = sorted(image_files)

if len(image_files) < 2:
    raise FileNotFoundError(
        "Please put at least TWO photographs inside the data folder."
    )

image1 = cv2.imread(str(image_files[0]))
image2 = cv2.imread(str(image_files[1]))

if image1 is None or image2 is None:
    raise ValueError("Unable to read input photographs.")

print("IMAGE 1:", image_files[0].name)
print("IMAGE 2:", image_files[1].name)


# ==========================================
# 3. CONVERT IMAGES TO GRAYSCALE
# ==========================================

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


# ==========================================
# 4. HOG FEATURE EXTRACTION
# ==========================================

def extract_hog(image):

    resized = cv2.resize(image, (128, 128))

    features, hog_image = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    return features, hog_image


hog_features1, hog_image1 = extract_hog(gray1)
hog_features2, hog_image2 = extract_hog(gray2)

print("\nHOG FEATURE EXTRACTION")
print("----------------------")
print("Image 1 HOG features:", len(hog_features1))
print("Image 2 HOG features:", len(hog_features2))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(hog_image1, cmap="gray")
plt.title("HOG Image 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image2, cmap="gray")
plt.title("HOG Image 2")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hog_features.png", dpi=300)
plt.close()


# ==========================================
# 5. SIFT KEYPOINT DETECTION
# ==========================================

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(
    gray1, None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2, None
)

print("\nSIFT FEATURE EXTRACTION")
print("-----------------------")
print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))


# Draw keypoints

sift_image1 = cv2.drawKeypoints(
    image1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

sift_image2 = cv2.drawKeypoints(
    image2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints_image1.jpg"),
    sift_image1
)

cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints_image2.jpg"),
    sift_image2
)


# ==========================================
# 6. FEATURE MATCHING FUNCTION
# ==========================================

def match_features(desc1, desc2):

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) < 2 or len(desc2) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return good_matches


# ==========================================
# 7. MATCH TWO ORIGINAL PHOTOGRAPHS
# ==========================================

good_matches = match_features(
    descriptors1,
    descriptors2
)

print("\nORIGINAL IMAGE MATCHING")
print("-----------------------")
print("Good matches:", len(good_matches))

matching_result = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "original_feature_matching.jpg"),
    matching_result
)


# ==========================================
# 8. ROTATE SECOND IMAGE
# ==========================================

height, width = gray2.shape

rotation_matrix = cv2.getRotationMatrix2D(
    (width // 2, height // 2),
    30,
    1.0
)

rotated_image = cv2.warpAffine(
    gray2,
    rotation_matrix,
    (width, height)
)

rotated_keypoints, rotated_descriptors = sift.detectAndCompute(
    rotated_image,
    None
)

rotated_matches = match_features(
    descriptors1,
    rotated_descriptors
)

print("\nROTATION EXPERIMENT")
print("-------------------")
print("Original matches:", len(good_matches))
print("Rotated matches:", len(rotated_matches))

rotated_result = cv2.drawMatches(
    image1,
    keypoints1,
    cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR),
    rotated_keypoints,
    rotated_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "rotated_feature_matching.jpg"),
    rotated_result
)

cv2.imwrite(
    str(OUTPUT_DIR / "rotated_image.jpg"),
    rotated_image
)


# ==========================================
# 9. RESIZE SECOND IMAGE
# ==========================================

resized_image = cv2.resize(
    gray2,
    None,
    fx=1.5,
    fy=1.5,
    interpolation=cv2.INTER_LINEAR
)

resized_keypoints, resized_descriptors = sift.detectAndCompute(
    resized_image,
    None
)

resized_matches = match_features(
    descriptors1,
    resized_descriptors
)

print("\nRESIZING EXPERIMENT")
print("-------------------")
print("Original matches:", len(good_matches))
print("Resized matches:", len(resized_matches))

resized_result = cv2.drawMatches(
    image1,
    keypoints1,
    cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR),
    resized_keypoints,
    resized_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "resized_feature_matching.jpg"),
    resized_result
)

cv2.imwrite(
    str(OUTPUT_DIR / "resized_image.jpg"),
    resized_image
)


# ==========================================
# 10. SAVE COMPARISON GRAPH
# ==========================================

labels = [
    "Original",
    "Rotated",
    "Resized"
]

match_counts = [
    len(good_matches),
    len(rotated_matches),
    len(resized_matches)
]

plt.figure(figsize=(8, 5))

plt.bar(labels, match_counts)

plt.title("SIFT Feature Matching Comparison")
plt.xlabel("Image Transformation")
plt.ylabel("Number of Good Matches")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "matching_comparison.png",
    dpi=300
)

plt.close()


# ==========================================
# 11. FINAL RESULTS
# ==========================================

print("\n" + "=" * 40)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 40)

print("HOG features extracted.")
print("SIFT keypoints detected.")
print("Original feature matching completed.")
print("Rotation experiment completed.")
print("Resizing experiment completed.")

print("\nOutputs saved in:")
print(OUTPUT_DIR)
