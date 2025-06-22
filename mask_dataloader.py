import os
import cv2
import torch

def load_image_masks(directory_path):
    """
    Loads all image masks from a given directory
    """
    image_masks = {}

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            # Load the image in grayscale mode
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                filename = str(os.path.basename(file_path).split('.')[0])

                image_masks[filename] = torch.tensor(img).to(torch.int)
            else:
                print(f"WARNING: Could not load mask with name {filename}")
        except Exception as e:
            print(f"ERROR loading mask with name {filename}: {e}")

    return image_masks
