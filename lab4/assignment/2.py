# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 19:23:28 2025

@author: USER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Load grayscale image ===
img = cv2.imread("Picture3.jpg", 0)  # Change path as needed
if img is None:
    raise ValueError("Image not found!")

# === Manually extract 5x5 top-left patch ===
img_patch = img[0:5, 0:5]  # Top-left patch

# === Determine gray levels dynamically and quantize patch ===
minp, maxp = img_patch.min(), img_patch.max()
lvls = int(maxp - minp + 1)
qpatch = (img_patch - minp).astype(np.uint8)

print("Original Patch:\n", img_patch)
print("Quantized Patch:\n", qpatch)
print("Number of Gray Levels:", lvls)

# === Manual GLCM functions ===
def manual_vertical_glcm_fn(image, levels=256):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h - 1): 
        for x in range(w):
            i = image[y, x]
            j = image[y + 1, x]
            glcm[i, j] += 1
    return glcm

def manual_horizontal_glcm_fn(image, levels=256):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h): 
        for x in range(w - 1):
            i = image[y, x]
            j = image[y, x + 1]
            glcm[i, j] += 1
    return glcm

def manual_diagonal_glcm_fn(image, levels=256):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h - 1): 
        for x in range(w - 1):
            i = image[y, x]
            j = image[y + 1, x + 1]
            glcm[i, j] += 1
    return glcm

# === Compute GLCMs ===
glcm_h = manual_horizontal_glcm_fn(qpatch, levels=lvls)
glcm_v = manual_vertical_glcm_fn(qpatch, levels=lvls)
glcm_d = manual_diagonal_glcm_fn(qpatch, levels=lvls)

# === Plot patch and GLCMs using only plt.subplot() ===
plt.figure(figsize=(20, 5))

# Plot patch
plt.subplot(1, 4, 1)
plt.imshow(qpatch, cmap='gray', interpolation='nearest')
plt.title("5x5 Image Patch")
plt.axis('off')

# Horizontal GLCM
plt.subplot(1, 4, 2)
im1 = plt.imshow(glcm_h, cmap='hot', interpolation='nearest')
plt.title("Horizontal GLCM")
plt.xlabel("Gray Level j")
plt.ylabel("Gray Level i")
plt.colorbar(im1, fraction=0.046, pad=0.04, label='Frequency')

# Vertical GLCM
plt.subplot(1, 4, 3)
im2 = plt.imshow(glcm_v, cmap='hot', interpolation='nearest')
plt.title("Vertical GLCM")
plt.xlabel("Gray Level j")
plt.ylabel("Gray Level i")
plt.colorbar(im2, fraction=0.046, pad=0.04, label='Frequency')

# Diagonal GLCM
plt.subplot(1, 4, 4)
im3 = plt.imshow(glcm_d, cmap='hot', interpolation='nearest')
plt.title("Diagonal GLCM")
plt.xlabel("Gray Level j")
plt.ylabel("Gray Level i")
plt.colorbar(im3, fraction=0.046, pad=0.04, label='Frequency')

plt.tight_layout()
plt.show()