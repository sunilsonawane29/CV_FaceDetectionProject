
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from pathlib import Path


# ============================================
# 1. SET INPUT AND OUTPUT PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    BASE_DIR
    / "data"
    / "WhatsApp Image 2026-11-02 at 6.20.35 PM.jpeg"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "unit3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 2. READ ORIGINAL PHOTOGRAPH
# ============================================

original = cv2.imread(str(IMAGE_PATH))

if original is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

# Resize for faster processing
original = cv2.resize(original, (600, 400))

# Save original image
cv2.imwrite(
    str(OUTPUT_DIR / "original_image.jpg"),
    original
)


# ============================================
# 3. ADD GAUSSIAN NOISE
# ============================================

np.random.seed(42)

gaussian_noise = np.random.normal(
    0,
    25,
    original.shape
).astype(np.float32)

noisy_gaussian = np.clip(
    original.astype(np.float32) + gaussian_noise,
    0,
    255
).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT_DIR / "gaussian_noise.jpg"),
    noisy_gaussian
)


# ============================================
# 4. ADD SALT AND PEPPER NOISE
# ============================================

noisy_sp = original.copy()

num_pixels = int(
    0.02 * original.shape[0] * original.shape[1]
)

# Add salt noise (white pixels)
rows = np.random.randint(
    0, original.shape[0], num_pixels
)

cols = np.random.randint(
    0, original.shape[1], num_pixels
)

noisy_sp[rows, cols] = 255

# Add pepper noise (black pixels)
rows = np.random.randint(
    0, original.shape[0], num_pixels
)

cols = np.random.randint(
    0, original.shape[1], num_pixels
)

noisy_sp[rows, cols] = 0

cv2.imwrite(
    str(OUTPUT_DIR / "salt_pepper_noise.jpg"),
    noisy_sp
)


# ============================================
# 5. APPLY GAUSSIAN AND MEDIAN FILTERS
# ============================================

# Experiment 1:
# Gaussian noise + Gaussian filter

restored_gaussian = cv2.GaussianBlur(
    noisy_gaussian,
    (5, 5),
    0
)


# Experiment 2:
# Gaussian noise + Median filter

gaussian_noisy_median = cv2.medianBlur(
    noisy_gaussian,
    5
)


# Experiment 3:
# Salt and pepper noise + Median filter

restored_median = cv2.medianBlur(
    noisy_sp,
    5
)


# Experiment 4:
# Salt and pepper noise + Gaussian filter

salt_pepper_gaussian = cv2.GaussianBlur(
    noisy_sp,
    (5, 5),
    0
)


# Save all filtered images

cv2.imwrite(
    str(OUTPUT_DIR / "gaussian_filtered.jpg"),
    restored_gaussian
)

cv2.imwrite(
    str(OUTPUT_DIR / "gaussian_noise_median_filtered.jpg"),
    gaussian_noisy_median
)

cv2.imwrite(
    str(OUTPUT_DIR / "median_filtered.jpg"),
    restored_median
)

cv2.imwrite(
    str(OUTPUT_DIR / "salt_pepper_gaussian_filtered.jpg"),
    salt_pepper_gaussian
)


# ============================================
# 6. CALCULATE PSNR AND SSIM
# ============================================

def calculate_metrics(reference, restored):

    psnr = peak_signal_noise_ratio(
        reference,
        restored,
        data_range=255
    )

    ssim = structural_similarity(
        reference,
        restored,
        channel_axis=2,
        data_range=255
    )

    return psnr, ssim


experiments = {
    "Gaussian Noise + Gaussian Filter": restored_gaussian,

    "Gaussian Noise + Median Filter": gaussian_noisy_median,

    "Salt-Pepper Noise + Median Filter": restored_median,

    "Salt-Pepper Noise + Gaussian Filter": salt_pepper_gaussian
}


print("\nIMAGE RESTORATION RESULTS")
print("=" * 45)

for name, restored_image in experiments.items():

    psnr, ssim = calculate_metrics(
        original,
        restored_image
    )

    print("\n" + name)
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


# ============================================
# 7. IMAGE THRESHOLDING
# ============================================

gray = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    str(OUTPUT_DIR / "binary_threshold.jpg"),
    binary
)


# ============================================
# 8. MORPHOLOGICAL OPERATIONS
# ============================================

kernel = np.ones(
    (5, 5),
    np.uint8
)

# Erosion
erosion = cv2.erode(
    binary,
    kernel,
    iterations=1
)

# Dilation
dilation = cv2.dilate(
    binary,
    kernel,
    iterations=1
)

# Opening
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

# Closing
closing = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)


# Save morphological outputs

cv2.imwrite(
    str(OUTPUT_DIR / "erosion.jpg"),
    erosion
)

cv2.imwrite(
    str(OUTPUT_DIR / "dilation.jpg"),
    dilation
)

cv2.imwrite(
    str(OUTPUT_DIR / "opening.jpg"),
    opening
)

cv2.imwrite(
    str(OUTPUT_DIR / "closing.jpg"),
    closing
)


# ============================================
# 9. CANNY EDGE DETECTION
# ============================================

edges = cv2.Canny(
    gray,
    100,
    200
)

cv2.imwrite(
    str(OUTPUT_DIR / "canny_edges.jpg"),
    edges
)


# ============================================
# 10. WATERSHED SEGMENTATION
# ============================================

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

# Otsu thresholding
_, thresh = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel_ws = np.ones(
    (3, 3),
    np.uint8
)

# Remove noise
opening_ws = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel_ws,
    iterations=2
)

# Sure background
sure_bg = cv2.dilate(
    opening_ws,
    kernel_ws,
    iterations=3
)

# Distance transformation
dist_transform = cv2.distanceTransform(
    opening_ws,
    cv2.DIST_L2,
    5
)

# Sure foreground
_, sure_fg = cv2.threshold(
    dist_transform,
    0.4 * dist_transform.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)

# Unknown region
unknown = cv2.subtract(
    sure_bg,
    sure_fg
)

# Connected components
_, markers = cv2.connectedComponents(
    sure_fg
)

markers = markers + 1

markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(
    original.copy(),
    markers
)

# Draw red watershed boundaries
watershed_result = original.copy()

watershed_result[markers == -1] = [
    0, 0, 255
]

cv2.imwrite(
    str(OUTPUT_DIR / "watershed_result.jpg"),
    watershed_result
)


# ============================================
# 11. DISPLAY AND SAVE COMPARISON RESULTS
# ============================================

images = [

    original,

    noisy_gaussian,

    restored_gaussian,

    gaussian_noisy_median,

    noisy_sp,

    restored_median,

    salt_pepper_gaussian,

    binary,

    erosion,

    dilation,

    opening,

    closing,

    edges,

    watershed_result
]


titles = [

    "Original",

    "Gaussian Noise",

    "Gaussian Filter",

    "Median on Gaussian Noise",

    "Salt and Pepper Noise",

    "Median Filter",

    "Gaussian on Salt-Pepper",

    "Thresholding",

    "Erosion",

    "Dilation",

    "Opening",

    "Closing",

    "Canny Edges",

    "Watershed"
]


plt.figure(figsize=(18, 15))

for i, (image, title) in enumerate(
    zip(images, titles)
):

    plt.subplot(4, 4, i + 1)

    if len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(
        title,
        fontsize=9
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    str(OUTPUT_DIR / "task1_results.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 12. FINAL MESSAGE
# ============================================

print("\n" + "=" * 45)

print("TASK 1 COMPLETED SUCCESSFULLY!")

print("=" * 45)

print(f"\nAll outputs saved in:\n{OUTPUT_DIR}")
