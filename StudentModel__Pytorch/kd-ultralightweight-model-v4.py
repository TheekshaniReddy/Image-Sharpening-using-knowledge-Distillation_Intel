#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# --- INSTALL LIBRARIES ---
# torchmetrics is a modern, efficient way to calculate metrics like SSIM on the GPU.
get_ipython().system('pip install scikit-image tqdm torchmetrics -q')

# --- IMPORT LIBRARIES ---
import os
import glob
from glob import glob
import random
import shutil
import time
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Use torchmetrics for efficient, GPU-accelerated SSIM calculation
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

print("✅ All libraries installed and imported successfully.")


# In[3]:


# # ==============================================================================
# #        CELL 2: CONFIGURATION (FOR "v8" - ARTIFACT REMOVAL MODEL)
# # ==============================================================================

# # ------------------------------------------------------------------------------
# # --- 1. PATHS (Update for your new dataset upload) ---
# # ------------------------------------------------------------------------------
# KAGGLE_DATASET_PATH = '/kaggle/input/ultralightweight-model-training-dataset-v8' 
# TEACHER_WEIGHTS_PATH = '/kaggle/input/ultralightweight-model-training-dataset-v8/final_best_model_precision-v3.pth'

# # ==============================================================================
# # --- AUTOMATIC PATHFINDER FUNCTION (No changes needed) ---
# # ==============================================================================
# def find_dataset_paths(base_path: str) -> dict:
#     found_paths = {'train': None, 'val': None}
#     print(f"Searching for 'train' and 'val' folders inside '{base_path}'...")
#     if not os.path.isdir(base_path): raise FileNotFoundError(f"The base path '{base_path}' does not exist.")
#     for root, dirs, files in os.walk(base_path):
#         if 'train' in dirs: found_paths['train'] = os.path.join(root, 'train')
#         if 'val' in dirs: found_paths['val'] = os.path.join(root, 'val')
#         if found_paths['train'] and found_paths['val']: break
#     if not found_paths['train']: raise FileNotFoundError(f"Could not find the 'train' directory inside {base_path}.")
#     if not found_paths['val']: print("⚠️ WARNING: Could not find the 'val' directory. Validation will be skipped.")
#     print(f"✅ Found train data at: {found_paths['train']}")
#     if found_paths['val']: print(f"✅ Found validation data at: {found_paths['val']}")
#     return found_paths

# try:
#     DATA_PATHS = find_dataset_paths(KAGGLE_DATASET_PATH)
# except FileNotFoundError as e:
#     print(e); raise

# # ==============================================================================
# # --- 2. TRAINING PARAMETERS ---
# # ==============================================================================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_EPOCHS = 63 # Let's keep your previous epoch count
# BATCH_SIZE = 32
# LEARNING_RATE = 1e-4

# # ==============================================================================
# # --- 3. RE-BALANCED LOSS WEIGHTS FOR AGGRESSIVE ARTIFACT REMOVAL ---
# # ==============================================================================
# ALPHA = 0.5    # Increased to prioritize clean ground truth
# GAMMA = 0.3    # No Change
# BETA  = 1.5    # Heavily increased to prioritize the clean Teacher output
# EPSILON = 0.4  # Slightly decreased edge priority
# LAMBDA = 0.02  # Heavily decreased to prevent sharpening of artifacts

# # ==============================================================================
# # --- 4. DATA & PATCH CONFIGURATION ---
# # ==============================================================================
# HR_PATCH_SIZE = 128
# DOWNSCALE_FACTOR = 4

# print(f"\nConfiguration loaded for ARTIFACT REMOVAL 'v8' model. Using device: {DEVICE}")
# print(f"All Loss Weights -> GT (ALPHA): {ALPHA}, VGG (GAMMA): {GAMMA}, Output KD (BETA): {BETA}, Gradient (EPSILON): {EPSILON}, FFT (LAMBDA): {LAMBDA}")

# ==============================================================================
#        CELL 2: CONFIGURATION (FOR FINAL "v9" RRDBNet MODEL)
# ==============================================================================

# ------------------------------------------------------------------------------
# --- 1. PATHS (Check that these are still correct) ---
# ------------------------------------------------------------------------------
KAGGLE_DATASET_PATH = '/kaggle/input/ultralightweight-model-training-dataset-v8' 
TEACHER_WEIGHTS_PATH = '/kaggle/input/ultralightweight-model-training-dataset-v8/final_best_model_precision-v3.pth'

# ==============================================================================
# --- AUTOMATIC PATHFINDER FUNCTION (No changes needed) ---
# ==============================================================================
def find_dataset_paths(base_path: str) -> dict:
    found_paths = {'train': None, 'val': None}
    print(f"Searching for 'train' and 'val' folders inside '{base_path}'...")
    if not os.path.isdir(base_path): raise FileNotFoundError(f"The base path '{base_path}' does not exist.")
    for root, dirs, files in os.walk(base_path):
        if 'train' in dirs: found_paths['train'] = os.path.join(root, 'train')
        if 'val' in dirs: found_paths['val'] = os.path.join(root, 'val')
        if found_paths['train'] and found_paths['val']: break
    if not found_paths['train']: raise FileNotFoundError(f"Could not find the 'train' directory inside {base_path}.")
    if not found_paths['val']: print("⚠️ WARNING: Could not find the 'val' directory. Validation will be skipped.")
    print(f"✅ Found train data at: {found_paths['train']}")
    if found_paths['val']: print(f"✅ Found validation data at: {found_paths['val']}")
    return found_paths

try:
    DATA_PATHS = find_dataset_paths(KAGGLE_DATASET_PATH)
except FileNotFoundError as e:
    print(e); raise

# ==============================================================================
# --- 2. TRAINING PARAMETERS ---
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 63
BATCH_SIZE = 32 # You might need to reduce this to 16 if you get memory errors with the new model
LEARNING_RATE = 1e-4

# ==============================================================================
# --- 3. FINAL LOSS WEIGHTS FOR RRDBNet ---
# ==============================================================================
ALPHA = 0.6    # Increased to keep the powerful model grounded to pixel truth
GAMMA = 0.4    # Perceptual loss is still very important for realism
BETA  = 1.4    # Strong guidance from the clean Teacher output remains critical
EPSILON = 0.5  # Gradient loss ensures sharp edges
LAMBDA = 0.1   # We can be slightly more aggressive with FFT loss as the model is smarter

# ==============================================================================
# --- 4. DATA & PATCH CONFIGURATION ---
# ==============================================================================
HR_PATCH_SIZE = 128
DOWNSCALE_FACTOR = 4

print(f"\nConfiguration loaded for FINAL 'v9' RRDBNet model. Using device: {DEVICE}")
print(f"All Loss Weights -> GT (ALPHA): {ALPHA}, VGG (GAMMA): {GAMMA}, Output KD (BETA): {BETA}, Gradient (EPSILON): {EPSILON}, FFT (LAMBDA): {LAMBDA}")


# In[4]:


# class KDTripletDataset(Dataset):
#     def __init__(self, root_dir, hr_patch_size=256, downscale_factor=4):
#         self.root_dir = root_dir
#         self.hr_patch_size = hr_patch_size
#         self.downscale_factor = downscale_factor
#         self.lr_patch_size = hr_patch_size // downscale_factor

#         self.hr_dir = os.path.join(root_dir, 'hr')
#         self.teacher_input_dir = os.path.join(root_dir, 'teacher_input')
#         self.student_input_dir = os.path.join(root_dir, 'student_input')

#         hr_glob_pattern = os.path.join(self.hr_dir, '**', '*')
#         self.image_files = sorted([
#             os.path.relpath(p, self.hr_dir) for p in glob(hr_glob_pattern, recursive=True) 
#             if p.lower().endswith(('.png', '.jpg', '.jpeg'))
#         ])
#         if not self.image_files: raise FileNotFoundError(f"No image files found in the subdirectories of {self.hr_dir}. Please check folder contents.")

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):
#         img_name = self.image_files[index]
#         hr_img = Image.open(os.path.join(self.hr_dir, img_name)).convert("RGB")
#         teacher_input_img = Image.open(os.path.join(self.teacher_input_dir, img_name)).convert("RGB")
#         student_input_img = Image.open(os.path.join(self.student_input_dir, img_name)).convert("RGB")

#         i, j, h, w = transforms.RandomCrop.get_params(hr_img, output_size=(self.hr_patch_size, self.hr_patch_size))
#         hr_patch = TF.crop(hr_img, i, j, h, w)
#         teacher_input_patch = TF.crop(teacher_input_img, i, j, h, w)
#         lr_i, lr_j, lr_h, lr_w = i // self.downscale_factor, j // self.downscale_factor, h // self.downscale_factor, w // self.downscale_factor
#         student_input_patch = TF.crop(student_input_img, lr_i, lr_j, lr_h, lr_w)

#         if random.random() > 0.5:
#             hr_patch, teacher_input_patch, student_input_patch = (TF.hflip(p) for p in [hr_patch, teacher_input_patch, student_input_patch])

#         to_tensor = transforms.ToTensor()
#         return {
#             'hr': to_tensor(hr_patch), 
#             'teacher_input': to_tensor(teacher_input_patch),
#             'student_input': to_tensor(student_input_patch)
#         }

# # --- Create DataLoaders using the automatically found paths ---
# train_dataset = KDTripletDataset(DATA_PATHS['train'], hr_patch_size=HR_PATCH_SIZE, downscale_factor=DOWNSCALE_FACTOR)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# # Create validation loader only if the path was found
# if DATA_PATHS['val']:
#     val_dataset = KDTripletDataset(DATA_PATHS['val'], hr_patch_size=HR_PATCH_SIZE, downscale_factor=DOWNSCALE_FACTOR)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
#     print(f"✅ DataLoaders created. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
# else:
#     val_loader = None # Set to None if no val data
#     print(f"✅ Training DataLoader created with {len(train_dataset)} samples. Validation is disabled.")

# ==============================================================================
#      CELL 3: DATASET CLASS WITH ADVANCED AUGMENTATION (FOR "v8" MODEL)
# ==============================================================================

class KDTripletDataset(Dataset):
    def __init__(self, root_dir, hr_patch_size=256, downscale_factor=4):
        self.root_dir = root_dir
        self.hr_patch_size = hr_patch_size
        self.downscale_factor = downscale_factor

        self.hr_dir = os.path.join(root_dir, 'hr')
        self.teacher_input_dir = os.path.join(root_dir, 'teacher_input')
        self.student_input_dir = os.path.join(root_dir, 'student_input')

        # --- NEW: Define a Color Jitter transform ---
        # This will be applied to the inputs to make the model robust to color shifts.
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        hr_glob_pattern = os.path.join(self.hr_dir, '**', '*')
        self.image_files = sorted([
            os.path.relpath(p, self.hr_dir) for p in glob(hr_glob_pattern, recursive=True) 
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not self.image_files: raise FileNotFoundError(f"No image files found in {self.hr_dir}.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        hr_img = Image.open(os.path.join(self.hr_dir, img_name)).convert("RGB")
        teacher_input_img = Image.open(os.path.join(self.teacher_input_dir, img_name)).convert("RGB")
        student_input_img = Image.open(os.path.join(self.student_input_dir, img_name)).convert("RGB")

        # --- 1. Synchronized Random Crop ---
        i, j, h, w = transforms.RandomCrop.get_params(hr_img, output_size=(self.hr_patch_size, self.hr_patch_size))
        hr_patch = TF.crop(hr_img, i, j, h, w)
        teacher_input_patch = TF.crop(teacher_input_img, i, j, h, w)
        lr_patch_size = self.hr_patch_size // self.downscale_factor
        student_input_patch = TF.crop(student_input_img, i // self.downscale_factor, j // self.downscale_factor, lr_patch_size, lr_patch_size)

        # --- 2. NEW: ADVANCED SYNCHRONIZED AUGMENTATIONS ---
        # Random Horizontal Flip
        if random.random() > 0.5:
            hr_patch, teacher_input_patch, student_input_patch = (TF.hflip(p) for p in [hr_patch, teacher_input_patch, student_input_patch])

        # Random Vertical Flip
        if random.random() > 0.5:
            hr_patch, teacher_input_patch, student_input_patch = (TF.vflip(p) for p in [hr_patch, teacher_input_patch, student_input_patch])

        # Random 90-degree Rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            hr_patch, teacher_input_patch, student_input_patch = (TF.rotate(p, angle) for p in [hr_patch, teacher_input_patch, student_input_patch])

        # Random Color Jitter (applied only to inputs, not the ground truth)
        # This teaches the model to handle color variations while aiming for correct HR colors.
        teacher_input_patch = self.color_jitter(teacher_input_patch)
        student_input_patch = self.color_jitter(student_input_patch)

        # --- 3. Convert to Tensor ---
        to_tensor = transforms.ToTensor()
        return {
            'hr': to_tensor(hr_patch), 
            'teacher_input': to_tensor(teacher_input_patch),
            'student_input': to_tensor(student_input_patch)
        }

# --- Create DataLoaders using the automatically found paths ---
# This part of the cell remains unchanged.
train_dataset = KDTripletDataset(DATA_PATHS['train'], hr_patch_size=HR_PATCH_SIZE, downscale_factor=DOWNSCALE_FACTOR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

if DATA_PATHS['val']:
    val_dataset = KDTripletDataset(DATA_PATHS['val'], hr_patch_size=HR_PATCH_SIZE, downscale_factor=DOWNSCALE_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"✅ DataLoaders created. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
else:
    val_loader = None
    print(f"✅ Training DataLoader created with {len(train_dataset)} samples. Validation is disabled.")


# In[5]:


# ==============================================================================
#        CELL 4: FINAL MODEL ARCHITECTURES ("v9" Student and Teacher)
# ==============================================================================

# ------------------------------------------------------------------------------
# --- 1. The Expert Teacher Model Architecture (Your original U-Net) ---
# ------------------------------------------------------------------------------
# We must define the Teacher's architecture so we can load its weights.
class Teacher_UNet_Wider(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=96):
        super(Teacher_UNet_Wider, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(features, features, 3, 1, 1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(nn.Conv2d(features, features * 2, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(features * 2, features * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.up_conv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.decoder1 = nn.Sequential(nn.Conv2d(features * 2, features, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(features, features, 3, 1, 1), nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(features, out_channels, 1, 1, 0)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1); b = self.bottleneck(p1)
        u1 = self.up_conv1(b); skip = torch.cat([u1, d1], dim=1); dec1 = self.decoder1(skip)
        residual_output = torch.tanh(self.final_conv(dec1))
        # The teacher returns its final residual and bottleneck features for potential use
        return residual_output, {'bottleneck': b}

# ------------------------------------------------------------------------------
# --- 2. The New, Intelligent Student Model Architecture (Lightweight RRDBNet) ---
# ------------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=32, gc=16):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1); self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1); self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1); self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1); self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x)); x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1))); x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1))); x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1))); x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=16):
        super(RRDB, self).__init__(); self.RDB1 = ResidualDenseBlock(nf, gc); self.RDB2 = ResidualDenseBlock(nf, gc); self.RDB3 = ResidualDenseBlock(nf, gc)
    def forward(self, x):
        out = self.RDB1(x); out = self.RDB2(out); out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet_v9(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4, gc=16, upscale=4):
        super(RRDBNet_v9, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc=gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upsampler = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(nf, nf, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(nf, nf, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk
        fea = self.upsampler(fea)
        fea = self.lrelu(self.conv_hr(fea))
        out = self.conv_last(fea)
        return torch.sigmoid(out)

print("✅ Final 'v9' RRDBNet and Teacher_UNet_Wider architectures defined.")


# In[6]:


# # --- 1. Perceptual (VGG) Loss ---
# class PerceptualLoss(nn.Module):
#     def __init__(self):
#         super(PerceptualLoss, self).__init__()
#         vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
#         self.features = nn.Sequential(*list(vgg.children())[:9]).eval().to(DEVICE)
#         for param in self.features.parameters(): param.requires_grad = False
#         self.loss_fn = nn.L1Loss()
#     def forward(self, student_output, ground_truth):
#         return self.loss_fn(self.features(student_output), self.features(ground_truth))

# # --- 2. Gradient (Edge) Loss ---
# class GradientLoss(nn.Module):
#     def __init__(self, device):
#         super(GradientLoss, self).__init__()
#         self.loss_fn = nn.L1Loss()
#         kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0).to(device)
#         kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0).to(device)
#         self.kernel_x = kernel_x.repeat(3, 1, 1, 1)
#         self.kernel_y = kernel_y.repeat(3, 1, 1, 1)
#     def forward(self, student_output, ground_truth):
#         grad_x_student = F.conv2d(student_output, self.kernel_x, padding='same', groups=3)
#         grad_y_student = F.conv2d(student_output, self.kernel_y, padding='same', groups=3)
#         grad_x_gt = F.conv2d(ground_truth, self.kernel_x, padding='same', groups=3)
#         grad_y_gt = F.conv2d(ground_truth, self.kernel_y, padding='same', groups=3)
#         return self.loss_fn(grad_x_student, grad_x_gt) + self.loss_fn(grad_y_student, grad_y_gt)

# # ==============================================================================
# # --- NEW: Frequency Domain (FFT) Loss for Aggressive Sharpness ---
# # ==============================================================================
# class FrequencyLoss(nn.Module):
#     def __init__(self, device):
#         super(FrequencyLoss, self).__init__()
#         self.loss_fn = nn.L1Loss()
#         self.device = device

#     def forward(self, student_output, ground_truth):
#         # Using torch.fft.rfft2 for real-valued inputs is more efficient
#         fft_student = torch.fft.rfft2(student_output, dim=(-2, -1))
#         fft_gt = torch.fft.rfft2(ground_truth, dim=(-2, -1))

#         # Compare the magnitude of the frequency components
#         magnitude_student = torch.abs(fft_student)
#         magnitude_gt = torch.abs(fft_gt)

#         return self.loss_fn(magnitude_student, magnitude_gt)

# # --- 3. Instantiate Models and ALL Losses ---
# student_model = StudentSuperResolutionNet(upscale_factor=DOWNSCALE_FACTOR)
# teacher_model = Teacher_UNet_Wider(features=96).to(DEVICE)

# task_loss_fn = nn.L1Loss()
# perceptual_loss_fn = PerceptualLoss()
# gradient_loss_fn = GradientLoss(device=DEVICE)
# frequency_loss_fn = FrequencyLoss(device=DEVICE) # Instantiate our new loss
# ssim_metric = SSIM(data_range=1.0).to(DEVICE)

# # --- 4. Multi-GPU Wrapper ---
# if torch.cuda.device_count() > 1:
#   print(f"✅ Using {torch.cuda.device_count()} GPUs for training via nn.DataParallel!")
#   student_model = nn.DataParallel(student_model)
# student_model.to(DEVICE)

# # --- 5. Load Teacher Weights and Freeze ---
# try:
#     teacher_model.load_state_dict(torch.load(TEACHER_WEIGHTS_PATH, map_location=DEVICE))
#     teacher_model.eval()
#     for param in teacher_model.parameters():
#         param.requires_grad = False
#     print("✅ Teacher model weights loaded successfully and model is frozen.")
# except Exception as e:
#     print(f"❌ ERROR loading teacher weights: {e}")

# # --- 6. Optimizer ---
# optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

# print("✅ All training components initialized, including new FrequencyLoss.")

# ==============================================================================
#       CELL 5: INITIALIZE THE FINAL RRDBNet MODEL AND LOSSES (WITH BUG FIX)
# ==============================================================================

# --- Define ALL Loss Functions ---
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.Sequential(*list(vgg.children())[:9]).eval().to(DEVICE)
        [p.requires_grad_(False) for p in self.features.parameters()]
        self.loss_fn = nn.L1Loss()
    def forward(self, so, gt):
        return self.loss_fn(self.features(so), self.features(gt))

class GradientLoss(nn.Module):
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        self.loss_fn = nn.L1Loss()

        # Define the Sobel kernels
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).float().unsqueeze(0).unsqueeze(0).to(device)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).float().unsqueeze(0).unsqueeze(0).to(device)

        # --- FIX IS HERE: Correctly create self.kernel_x and self.kernel_y ---
        self.kernel_x = kernel_x.repeat(3, 1, 1, 1)
        self.kernel_y = kernel_y.repeat(3, 1, 1, 1) # This was the missing part

    def forward(self, so, gt):
        # --- FIX IS HERE: Use the correct variable names ---
        grad_x_student = F.conv2d(so, self.kernel_x, padding='same', groups=3)
        grad_y_student = F.conv2d(so, self.kernel_y, padding='same', groups=3) # Changed from self.ky

        grad_x_gt = F.conv2d(gt, self.kernel_x, padding='same', groups=3)
        grad_y_gt = F.conv2d(gt, self.kernel_y, padding='same', groups=3)

        return self.loss_fn(grad_x_student, grad_x_gt) + self.loss_fn(grad_y_student, grad_y_gt)

class FrequencyLoss(nn.Module):
    def __init__(self, device):
        super(FrequencyLoss, self).__init__()
        self.loss_fn = nn.L1Loss()
        self.device = device
    def forward(self, so, gt):
        fft_student = torch.fft.rfft2(so, dim=(-2,-1))
        fft_gt = torch.fft.rfft2(gt, dim=(-2,-1))
        return self.loss_fn(torch.abs(fft_student), torch.abs(fft_gt))

# --- Instantiate Models and Losses ---
student_model = RRDBNet_v9(nf=32, nb=4, gc=16, upscale=DOWNSCALE_FACTOR)
teacher_model = Teacher_UNet_Wider(features=96).to(DEVICE)

task_loss_fn = nn.L1Loss()
perceptual_loss_fn = PerceptualLoss()
gradient_loss_fn = GradientLoss(device=DEVICE)
frequency_loss_fn = FrequencyLoss(device=DEVICE)
ssim_metric = SSIM(data_range=1.0).to(DEVICE)

# --- Multi-GPU Wrapper ---
if torch.cuda.device_count() > 1:
    print(f"✅ Using {torch.cuda.device_count()} GPUs!")
    student_model = nn.DataParallel(student_model)
student_model.to(DEVICE)

# --- Load Teacher Weights and Freeze ---
try:
    teacher_model.load_state_dict(torch.load(TEACHER_WEIGHTS_PATH, map_location=DEVICE))
    teacher_model.eval()
    [p.requires_grad_(False) for p in teacher_model.parameters()]
    print("✅ Teacher model loaded and frozen.")
except Exception as e:
    print(f"❌ ERROR loading teacher weights: {e}")

# --- Optimizer ---
optimizer = optim.Adam(student_model.parameters(), lr=LEARNING_RATE)
print("✅ All training components for the final 'v9' model initialized.")


# In[7]:


# history = {'train_loss': [], 'val_ssim': []}
# best_val_ssim = 0.0

# print("\n" + "="*50)
# print("🚀 Starting Training with FFT Loss for Maximum Sharpness 🚀")
# print("="*50 + "\n")

# for epoch in range(NUM_EPOCHS):
#     # --- Training Phase ---
#     student_model.train()
#     running_train_loss = 0.0
#     loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

#     for batch in loop:
#         student_input = batch['student_input'].to(DEVICE)
#         teacher_input = batch['teacher_input'].to(DEVICE)
#         hr_patches = batch['hr'].to(DEVICE)

#         optimizer.zero_grad()

#         # --- Forward Pass ---
#         student_outputs, student_features = student_model(student_input)

#         with torch.no_grad():
#             teacher_residual, teacher_features = teacher_model(teacher_input)
#             teacher_outputs = (teacher_input + teacher_residual).clamp(0, 1)

#         # --- Loss Calculation with NEW Frequency Loss ---
#         loss_gt = task_loss_fn(student_outputs, hr_patches)
#         loss_perceptual = perceptual_loss_fn(student_outputs, hr_patches)
#         loss_distill_output = task_loss_fn(student_outputs, teacher_outputs.detach())
#         loss_gradient = gradient_loss_fn(student_outputs, hr_patches)
#         loss_frequency = frequency_loss_fn(student_outputs, hr_patches) # Calculate the new FFT loss

#         # --- Combine all losses ---
#         total_loss = (ALPHA * loss_gt) + \
#                      (GAMMA * loss_perceptual) + \
#                      (BETA  * loss_distill_output) + \
#                      (EPSILON * loss_gradient) + \
#                      (LAMBDA * loss_frequency) # Add the powerful new loss

#         total_loss.backward()
#         optimizer.step()

#         running_train_loss += total_loss.item()
#         loop.set_postfix(loss=total_loss.item())

#     epoch_train_loss = running_train_loss / len(train_loader)
#     history['train_loss'].append(epoch_train_loss)

#     # --- Validation Phase ---
#     student_model.eval()
#     running_val_ssim = 0.0
#     if val_loader:
#         with torch.no_grad():
#             for batch in val_loader:
#                 student_input = batch['student_input'].to(DEVICE)
#                 hr_patches = batch['hr'].to(DEVICE)
#                 student_outputs, _ = student_model(student_input)
#                 student_outputs = student_outputs.clamp(0, 1)
#                 running_val_ssim += ssim_metric(student_outputs, hr_patches)

#         epoch_val_ssim = running_val_ssim / len(val_loader)
#         history['val_ssim'].append(epoch_val_ssim.item())
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {epoch_train_loss:.5f}, Val SSIM: {epoch_val_ssim:.4f}")

#         if epoch_val_ssim > best_val_ssim:
#             best_val_ssim = epoch_val_ssim
#             # Save the new model with a descriptive name
#             save_path = '/kaggle/working/best_fft_sharp_model.pth'
#             if isinstance(student_model, nn.DataParallel):
#                 torch.save(student_model.module.state_dict(), save_path)
#             else:
#                 torch.save(student_model.state_dict(), save_path)
#             print(f"----> 🏆 New best model saved with Val SSIM: {best_val_ssim:.4f}")
#     else:
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {epoch_train_loss:.5f} (Validation Skipped)")

# print("\n✅ Training finished.")

# ==============================================================================
#        CELL 6: FINAL TRAINING LOOP FOR "v9" RRDBNet MODEL
# ==============================================================================

history = {'train_loss': [], 'val_ssim': []}
best_val_ssim = 0.0
print("\n" + "="*50); print("🚀 Starting FINAL Training Run (v9 RRDBNet) 🚀"); print("="*50 + "\n")

for epoch in range(NUM_EPOCHS):
    student_model.train()
    running_train_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

    for batch in loop:
        student_input, teacher_input, hr_patches = batch['student_input'].to(DEVICE), batch['teacher_input'].to(DEVICE), batch['hr'].to(DEVICE)
        optimizer.zero_grad()

        # --- Simplified Forward Pass ---
        student_outputs = student_model(student_input)

        with torch.no_grad():
            teacher_residual, _ = teacher_model(teacher_input)
            teacher_outputs = (teacher_input + teacher_residual).clamp(0, 1)

        # --- Final Loss Calculation (No Feature Loss) ---
        loss_gt = task_loss_fn(student_outputs, hr_patches)
        loss_perceptual = perceptual_loss_fn(student_outputs, hr_patches)
        loss_distill_output = task_loss_fn(student_outputs, teacher_outputs.detach())
        loss_gradient = gradient_loss_fn(student_outputs, hr_patches)
        loss_frequency = frequency_loss_fn(student_outputs, hr_patches)

        total_loss = (ALPHA * loss_gt) + (GAMMA * loss_perceptual) + (BETA  * loss_distill_output) + (EPSILON * loss_gradient) + (LAMBDA * loss_frequency)

        total_loss.backward()
        optimizer.step()
        running_train_loss += total_loss.item()
        loop.set_postfix(loss=total_loss.item())

    history['train_loss'].append(running_train_loss / len(train_loader))

    # --- Validation Phase ---
    student_model.eval()
    running_val_ssim = 0.0
    if val_loader:
        with torch.no_grad():
            for batch in val_loader:
                student_input, hr_patches = batch['student_input'].to(DEVICE), batch['hr'].to(DEVICE)
                student_outputs = student_model(student_input).clamp(0, 1)
                running_val_ssim += ssim_metric(student_outputs, hr_patches)

        epoch_val_ssim = running_val_ssim / len(val_loader)
        history['val_ssim'].append(epoch_val_ssim.item())
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {history['train_loss'][-1]:.5f}, Val SSIM: {epoch_val_ssim:.4f}")

        if epoch_val_ssim > best_val_ssim:
            best_val_ssim = epoch_val_ssim
            save_path = '/kaggle/working/best_v9_RRDB_model.pth'
            if isinstance(student_model, nn.DataParallel): torch.save(student_model.module.state_dict(), save_path)
            else: torch.save(student_model.state_dict(), save_path)
            print(f"----> 🏆 New best model saved with Val SSIM: {best_val_ssim:.4f}")
    else:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {history['train_loss'][-1]:.5f} (Validation Skipped)")

print("\n✅ Final training finished.")


# In[8]:


# ==============================================================================
#             CELL 7: FINAL MODEL EVALUATION (SELF-CONTAINED)
# ==============================================================================

# --- This cell is designed to be self-contained. It includes all necessary imports and definitions. ---

# ------------------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------------------
import os
import time
from collections import OrderedDict
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import and instantiate torchmetrics for SSIM calculation
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
ssim_metric = SSIM(data_range=1.0).to(DEVICE) # Assumes DEVICE is defined from a previous cell

print("✅ Evaluation script setup complete.")

# ------------------------------------------------------------------------------
# 2. LOAD THE BEST TRAINED STUDENT MODEL
# ------------------------------------------------------------------------------
print("Loading the best student model for final evaluation...")

# --- UPDATE THIS PATH if you saved the model with a different name ---
MODEL_TO_EVALUATE = '/kaggle/working/best_fft_sharp_model.pth'

# Instantiate the correct StudentSuperResolutionNet architecture
# Assumes DOWNSCALE_FACTOR and DEVICE are available from Cell 2
final_student_model = StudentSuperResolutionNet(upscale_factor=DOWNSCALE_FACTOR).to(DEVICE)

# --- Clean the state dictionary before loading ---
try:
    state_dict = torch.load(MODEL_TO_EVALUATE, map_location=DEVICE)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.`
        new_state_dict[name] = v
    final_student_model.load_state_dict(new_state_dict)
    final_student_model.eval()
    print("✅ Best student model loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found at '{MODEL_TO_EVALUATE}'.")
    print("   Please check the filename in this script and ensure training completed successfully.")
except Exception as e:
    print(f"❌ An error occurred while loading the model: {e}")


# ------------------------------------------------------------------------------
# 3. CREATE THE BENCHMARK DATASET (for full images)
# ------------------------------------------------------------------------------
class BenchmarkDataset(Dataset):
    def __init__(self, root_dir):
        self.degraded_dir = os.path.join(root_dir, 'teacher_input')
        self.hr_dir = os.path.join(root_dir, 'hr')
        glob_pattern = os.path.join(self.hr_dir, '**', '*')
        self.image_files = sorted([os.path.relpath(p, self.hr_dir) for p in glob(glob_pattern, recursive=True) if p.lower().endswith(('.png', '.jpg', '.jpeg'))])
    def __len__(self): return len(self.image_files)
    def __getitem__(self, index):
        img_name = self.image_files[index]
        degraded_img = Image.open(os.path.join(self.degraded_dir, img_name)).convert("RGB")
        hr_img = Image.open(os.path.join(self.hr_dir, img_name)).convert("RGB")
        return transforms.ToTensor()(degraded_img), transforms.ToTensor()(hr_img)

# Assumes DATA_PATHS is available from Cell 2
if 'DATA_PATHS' in locals() and DATA_PATHS.get('val'):
    benchmark_dataset = BenchmarkDataset(DATA_PATHS['val'])
    print(f"✅ Created benchmark dataset with {len(benchmark_dataset)} full-resolution images.")
else:
    benchmark_dataset = None
    print("⚠️ Benchmark evaluation skipped: Validation data not found.")


# ------------------------------------------------------------------------------
# 4. RUN THE EVALUATION LOOP
# ------------------------------------------------------------------------------
if benchmark_dataset:
    total_ssim = 0
    total_time = 0
    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        for i in tqdm(range(len(benchmark_dataset)), desc="Evaluating Benchmark"):
            degraded_tensor, hr_tensor = benchmark_dataset[i]

            start_time = time.time()

            # This logic is correct and handles all our previous fixes
            student_input_size = (hr_tensor.shape[2] // DOWNSCALE_FACTOR, hr_tensor.shape[1] // DOWNSCALE_FACTOR)
            degraded_pil = to_pil(degraded_tensor)
            student_input_pil = degraded_pil.resize(student_input_size, Image.BICUBIC)
            student_input_tensor = transforms.ToTensor()(student_input_pil).unsqueeze(0).to(DEVICE)

            final_output_tensor, _ = final_student_model(student_input_tensor)
            final_output_tensor = final_output_tensor.clamp(0, 1)

            end_time = time.time()
            total_time += (end_time - start_time)

            # Crop the HR ground truth to match the output size for a fair comparison
            output_h, output_w = final_output_tensor.shape[2], final_output_tensor.shape[3]
            hr_tensor_batch = hr_tensor.unsqueeze(0).to(DEVICE)
            hr_tensor_cropped = hr_tensor_batch[:, :, :output_h, :output_w]

            total_ssim += ssim_metric(final_output_tensor, hr_tensor_cropped).item()

    # --- 5. DISPLAY FINAL RESULTS ---
    avg_ssim = total_ssim / len(benchmark_dataset)
    avg_fps = len(benchmark_dataset) / total_time

    print("\n" + "="*40); print("🏁 Final Evaluation Results 🏁"); print("="*40)
    print(f"Average SSIM on Benchmark Set: {avg_ssim:.4f}")
    print(f"Average FPS (Frames Per Second): {avg_fps:.2f}")
    print("="*40)

    # --- 6. VISUALIZE A FINAL RESULT ---
    final_output_pil = to_pil(final_output_tensor.squeeze(0).cpu())
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(degraded_pil); ax[0].set_title('Degraded Input'); ax[0].axis('off')
    ax[1].imshow(final_output_pil); ax[1].set_title(f'Student Output (SSIM: {avg_ssim:.4f})'); ax[1].axis('off')
    ax[2].imshow(to_pil(hr_tensor)); ax[2].set_title('Ground Truth'); ax[2].axis('off')
    plt.show()

