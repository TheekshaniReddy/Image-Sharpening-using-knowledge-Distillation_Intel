
import io
import os
import random
import shutil
from glob import glob
from typing import Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# Update these paths for your local machine
SOURCE_DIR: str = "path_to_raw_hr_images"
TARGET_DIR: str = "path_to_save_triplets" # Use a new folder for this robust dataset
DOWNSCALE_FACTOR: int = 4
ALLOWED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
RANDOM_SEED = 42
TRAIN_SPLIT_RATIO = 0.9

# ==============================================================================
# --- Robust Core Degradation Logic 
# ==============================================================================

def create_degradation_pair_robust(hr_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    Applies a MORE AGGRESSIVE and VARIED degradation pipeline to teach the model
    how to handle severe blockiness, artifacts, and noise.
    """
    img = hr_image.convert("RGB")
    orig_w, orig_h = img.size

    resampling_method = random.choice([Image.BICUBIC, Image.NEAREST])
    small_size = (max(1, orig_w // DOWNSCALE_FACTOR), max(1, orig_h // DOWNSCALE_FACTOR))
    downscaled = img.resize(small_size, resample=resampling_method)

    JPEG_QUALITY_RANGE_ROBUST = (5, 35)
    with io.BytesIO() as buf:
        random_jpeg_quality = random.randint(JPEG_QUALITY_RANGE_ROBUST[0], JPEG_QUALITY_RANGE_ROBUST[1])
        downscaled.save(buf, format="JPEG", quality=random_jpeg_quality)
        buf.seek(0)
        student_input_image = Image.open(buf).convert("RGB")
    
    if random.random() > 0.5:
        student_np = np.array(student_input_image).astype(np.float32)
        noise_strength = random.uniform(0, 10)
        noise = np.random.normal(0, noise_strength, student_np.shape)
        student_np = np.clip(student_np + noise, 0, 255)
        student_input_image = Image.fromarray(student_np.astype(np.uint8))

    teacher_input_image = student_input_image.resize((orig_w, orig_h), resample=Image.BICUBIC)
    return teacher_input_image, student_input_image

# ==============================================================================
# --- Main Script Logic 
# ==============================================================================

def process_and_save_triplet(hr_path: str, split_name: str):
    """Processes one HR image and saves the full triplet."""
    try:
        rel_path = os.path.relpath(hr_path, SOURCE_DIR)
        
        # Define save paths for all three components
        hr_save_path = os.path.join(TARGET_DIR, split_name, 'hr', rel_path)
        teacher_save_path = os.path.join(TARGET_DIR, split_name, 'teacher_input', rel_path)
        student_save_path = os.path.join(TARGET_DIR, split_name, 'student_input', rel_path)

        
        # We must ensure the parent directory for EACH output file exists before saving.
        os.makedirs(os.path.dirname(hr_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(teacher_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(student_save_path), exist_ok=True)

        # 1. Save HR Ground Truth
        shutil.copy(hr_path, hr_save_path)

        # 2. Generate and save the Teacher and Student inputs
        with Image.open(hr_path) as hr_img:
            teacher_img, student_img = create_degradation_pair_robust(hr_img)
            
            teacher_img.save(teacher_save_path, format="PNG")
            student_img.save(student_save_path, format="PNG")
            
    except Exception as e:
        print(f"Warning: Failed to process {hr_path}. Error: {e}")

if __name__ == "__main__":
    print("Starting ROBUST Knowledge Distillation triplet preparation...")
    random.seed(RANDOM_SEED)

    for split in ['train', 'val']:
        for folder in ['hr', 'teacher_input', 'student_input']:
            os.makedirs(os.path.join(TARGET_DIR, split, folder), exist_ok=True)

    all_images = sorted([p for p in glob(os.path.join(SOURCE_DIR, '', '*'), recursive=True) if p.lower().endswith(ALLOWED_EXTENSIONS)])
    random.shuffle(all_images)
    train_images, val_images = all_images[:int(len(all_images) * TRAIN_SPLIT_RATIO)], all_images[int(len(all_images) * TRAIN_SPLIT_RATIO):]

    print(f"Found {len(all_images)} images. Splitting into {len(train_images)} train and {len(val_images)} val.")

    for path in tqdm(train_images, desc="Processing Train Set"): process_and_save_triplet(path, 'train')
    for path in tqdm(val_images, desc="Processing Val Set"): process_and_save_triplet(path, 'val')
        
    print(f"\nRobust triplet preparation complete. Dataset saved to '{TARGET_DIR}'.")