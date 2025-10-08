# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 19:22:57 2025

@author: USER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_glcm(glcm):
    s = np.sum(glcm)
    return glcm / s if s > 0 else glcm

def max_probability(glcm):
    norm = normalize_glcm(glcm)
    return np.max(norm)

def energy(glcm):
    norm = normalize_glcm(glcm)
    return np.sum(norm ** 2)

def entropy(glcm):
    norm = normalize_glcm(glcm)
    nz = norm[norm > 0]
    return -np.sum(nz * np.log2(nz))

def contrast(glcm):
    norm = normalize_glcm(glcm)
    i, j = np.indices(norm.shape)
    return np.sum((i - j) ** 2 * norm)

def homogeneity(glcm):
    norm = normalize_glcm(glcm)
    i, j = np.indices(norm.shape)
    return np.sum(norm / (1.0 + np.abs(i - j)))


def manual_horizontal_glcm(image, levels):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h):
        for x in range(w - 1):
            i, j = image[y, x], image[y, x + 1]
            glcm[i, j] += 1
    return glcm

def manual_vertical_glcm(image, levels):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h - 1):
        for x in range(w):
            i, j = image[y, x], image[y + 1, x]
            glcm[i, j] += 1
    return glcm

def manual_diagonal_glcm(image, levels):
    glcm = np.zeros((levels, levels), dtype=int)
    h, w = image.shape
    for y in range(h - 1):
        for x in range(w - 1):
            i, j = image[y, x], image[y + 1, x + 1]
            glcm[i, j] += 1
    return glcm


def process_image(img):
    levels = 256
    
    glcm_h = manual_horizontal_glcm(img, levels)
    glcm_v = manual_vertical_glcm(img, levels)
    glcm_d = manual_diagonal_glcm(img, levels)
    
    
    for name, glcm in zip(["Horizontal", "Vertical", "Diagonal"], [glcm_h, glcm_v, glcm_d]):
        print(f"\n{name} GLCM Features:")
        print(f"  Max Probability: {max_probability(glcm):.4f}")
        print(f"  Energy         : {energy(glcm):.4f}")
        print(f"  Entropy        : {entropy(glcm):.4f}")
        print(f"  Contrast       : {contrast(glcm):.4f}")
        print(f"  Homogeneity    : {homogeneity(glcm):.4f}")
    

    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title("Original-Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    im1 = plt.imshow(glcm_h, cmap='hot', interpolation='nearest')
    plt.title("Horizontal GLCM")
    plt.xlabel("Gray Level j")
    plt.ylabel("Gray Level i")
    
    plt.subplot(1, 4, 3)
    im2 = plt.imshow(glcm_v, cmap='hot', interpolation='nearest')
    plt.title("Vertical GLCM")
    plt.xlabel("Gray Level j")
    plt.ylabel("Gray Level i")

    
    plt.subplot(1, 4, 4)
    im3 = plt.imshow(glcm_d, cmap='hot', interpolation='nearest')
    plt.title("Diagonal GLCM")
    plt.xlabel("Gray Level j")
    plt.ylabel("Gray Level i")

    plt.tight_layout()
    plt.show()

    
img1 = cv2.imread("Picture1.jpg", cv2.IMREAD_GRAYSCALE)    
img2 = cv2.imread("Picture2.jpg", cv2.IMREAD_GRAYSCALE)   
img3 = cv2.imread("Picture3.jpg", cv2.IMREAD_GRAYSCALE)
img4=cv2.imread("1.jpg",cv2.IMREAD_GRAYSCALE)
img5=cv2.imread("2.jpg",cv2.IMREAD_GRAYSCALE)
process_image(img1)
process_image(img2)
process_image(img3)
process_image(img4)
