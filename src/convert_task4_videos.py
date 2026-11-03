plt.show()
from pathlib import Path
import subprocess
import imageio_ffmpeg

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "outputs" / "unit5" / "task4"

# Locate FFmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

videos = [
    "frame_differencing.mp4",
    "gmm_background_subtraction.mp4",
    "lucas_kanade_optical_flow.mp4"
]

for filename in videos:
    input_path = VIDEO_DIR / filename
    output_path = VIDEO_DIR / (input_path.stem + "_chrome.mp4")

    if not input_path.exists():
        print("Video not found:", input_path)
        continue

    command = [
        ffmpeg,
        "-y",
        "-i", str(input_path),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path)
    ]

    print("\nConverting:", filename)

    try:
        subprocess.run(command, check=True)
        print("Successfully created:", output_path.name)
    except subprocess.CalledProcessError:
        print("Conversion failed:", filename)

print("\nVIDEO CONVERSION COMPLETED!")
