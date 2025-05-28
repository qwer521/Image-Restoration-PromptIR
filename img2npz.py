import argparse
import os
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Convert image folder to .npz archive"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing images",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default="pred.npz",
        help="Output .npz filename",
    )
    args = parser.parse_args()

    images_dict = {}
    for filename in os.listdir(args.folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(args.folder_path, filename)
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img).transpose(2, 0, 1)  # (3, H, W)
            images_dict[filename] = arr

    np.savez(args.output_npz, **images_dict)
    print(
        f"Saved {len(images_dict)} images from \
        '{args.folder_path}' to '{args.output_npz}'"
    )


if __name__ == "__main__":
    main()
