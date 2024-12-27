import cv2
import os
import re
import argparse


def natural_sort_key(s):
    """Helper function to generate a natural sort key."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def create_video_from_folders(folder1, folder2, output_video, frame_rate=1):
    # Get lists of image files from both folders
    images1 = [
        os.path.join(folder1, img)
        for img in os.listdir(folder1)
        if img.endswith(".jpg")
    ]
    images2 = [
        os.path.join(folder2, img)
        for img in os.listdir(folder2)
        if img.endswith(".jpg")
    ]

    # Apply natural sorting to the image lists
    images1.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    images2.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

    # Ensure both folders have the same number of images
    if len(images1) != len(images2):
        raise ValueError("The number of images in both folders must be the same.")

    # Read the first image to determine the dimensions
    img1 = cv2.imread(images1[0])
    img2 = cv2.imread(images2[0])

    if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1]:
        raise ValueError("Images in both folders must have the same dimensions.")

    height, width, _ = img1.shape
    combined_width = width * 2

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, frame_rate, (combined_width, height))

    # Combine images and write to video
    for img1_path, img2_path in zip(images1, images2):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Combine images side by side
        combined_img = cv2.hconcat([img1, img2])

        # Write the frame to the video
        out.write(combined_img)

    # Release the VideoWriter
    out.release()
    print(f"Video saved as {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine images from two folders into a side-by-side video."
    )
    parser.add_argument(
        "--folder1", required=True, help="Path to the first folder containing images."
    )
    parser.add_argument(
        "--folder2", required=True, help="Path to the second folder containing images."
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output video file."
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=1,
        help="Frame rate for the video (default: 1).",
    )

    args = parser.parse_args()

    create_video_from_folders(args.folder1, args.folder2, args.output, args.frame_rate)
