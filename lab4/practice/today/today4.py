# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 07:10:21 2025

@author: USER
"""

import cv2
import numpy as np
from math import pi
from tabulate import tabulate
import math
from math import sqrt

# -------------------------------
# Functions
# -------------------------------

def find_max_d(binary_image):
    min_x = min_y = 1e9
    max_x = max_y = 0
    h, w = binary_image.shape
    for x in range(h):
        for y in range(w):
            if binary_image[x, y] > 0:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    return max(max_x - min_x, max_y - min_y)

def calculate_descriptors(binary_image):
    # Erode to find border
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    border_image = binary_image - eroded
    

    # Show images
    cv2.imshow("image - Binary", binary_image)
    cv2.imshow("image - Border", border_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Shape properties
    area = np.count_nonzero(binary_image)
    perimeter = np.count_nonzero(border_image)
    max_d = find_max_d(binary_image)

    compact = (perimeter**2) / area if area > 0 else 0
    form_fact = (4 * pi * area) / (perimeter**2) if perimeter > 0 else 0
    roundness = (4 * area) / (pi * max_d**2) if max_d > 0 else 0

    return np.array([form_fact, roundness, compact])  # order = [FF, R, C]

def cosine_similarity(v1, v2):
    # dot product
    dot = sum(v1[i] * v2[i] for i in range(len(v1)))
    # manual norms
    norm1 = sqrt(sum(v1[i]**2 for i in range(len(v1))))
    norm2 = sqrt(sum(v2[i]**2 for i in range(len(v2))))
    # cosine similarity
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0


# -------------------------------
# Main Code
# -------------------------------

image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png','st.jpg']

descriptors = []
for name in image_name:
    img_gray = cv2.imread(name, 0)

    descriptors.append(calculate_descriptors(img_gray))

# Training descriptors (first 3)
train = descriptors[:3]
# Test descriptors (rest)
test = descriptors[3:]

# Cosine similarity matrix
similarity = []
for t in test:
    row = [cosine_similarity(tr, t) for tr in train]
    similarity.append(row)

# Print descriptors
print("\n=== Shape Descriptors (FF, Roundness, Compactness) ===")
print(tabulate(descriptors,
               headers=["Form Factor","Roundness","Compactness"],
               showindex=image_name,
               tablefmt="grid"))

# Print similarity matrix
print("\n=== Cosine Similarity Matrix ===")
row_headers = image_name[3:]
col_headers = image_name[:3]
print(tabulate(similarity, headers=col_headers, showindex=row_headers, tablefmt='grid'))
