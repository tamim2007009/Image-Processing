# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:04:35 2025

@author: USER
"""

import cv2
import numpy as np
from math import pi, sqrt, log
from tabulate import tabulate

# -------------------------------------------------------
# Find maximum distance (bounding box diagonal)
# -------------------------------------------------------
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

# -------------------------------------------------------
# Calculate shape descriptors (with eccentricity from contour)
# -------------------------------------------------------
def calculate_descriptors(binary_image):
    # Convert to binary
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Erode to find border
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    border_image = binary_image - eroded

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (0, 0, 0, 0)
    cnt = max(contours, key=cv2.contourArea)  # largest contour

    # Optional visualization
    cv2.imshow("Binary", binary_image)
    cv2.drawContours(binary_image, [cnt], -1, 127, 2)
    cv2.imshow("Contour", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Basic shape properties
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    max_d = find_max_d(binary_image)

    # Shape descriptors
    compact = (perimeter**2) / area if area > 0 else 0
    form_fact = (4 * pi * area) / (perimeter**2) if perimeter > 0 else 0
    roundness = (4 * area) / (pi * max_d**2) if max_d > 0 else 0

    # Eccentricity from fitted ellipse
    if len(cnt) >= 5:  # fitEllipse requires at least 5 points
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        a = max(MA, ma) / 2.0  # major axis radius
        b = min(MA, ma) / 2.0  # minor axis radius
        if a != 0:
            eccentricity = sqrt(1 - (b**2 / a**2))
        else:
            eccentricity = 0
    else:
        eccentricity = 0

    return (compact, form_fact, roundness, eccentricity)

# -------------------------------------------------------
# Distance metrics
# -------------------------------------------------------
def euclidean_dist(t1, t2):
    return sqrt(sum((np.array(t1) - np.array(t2))**2))

def cosine_dist(t1, t2):
    t1 = np.array(t1)
    t2 = np.array(t2)
    dot = np.dot(t1, t2)
    norm = np.linalg.norm(t1) * np.linalg.norm(t2)
    return 1 - (dot / norm) if norm != 0 else 1

def kl_divergence(t1, t2):
    t1 = np.array(t1, dtype=np.float64)
    t2 = np.array(t2, dtype=np.float64)
    t1 = np.clip(t1, 1e-10, None)
    t2 = np.clip(t2, 1e-10, None)
    return np.sum(t1 * np.log(t1 / t2))

# -------------------------------------------------------
# Main Code
# -------------------------------------------------------
image_name = ['c1.jpg', 't1.jpg', 'p1.png', 'c2.jpg', 't2.jpg', 'p2.png']

descriptors = []
for name in image_name:
    img_gray = cv2.imread(name, 0)
    if img_gray is None:
        print(f"Error: Image {name} not found.")
        descriptors.append((0, 0, 0, 0))
        continue
    descriptors.append(calculate_descriptors(img_gray))

# Training descriptors (first 3)
train = descriptors[:3]

# Distance matrices
euclid_dist = []
cos_dist = []
kl_dist = []

for i in range(3, len(image_name)):
    desc = descriptors[i]
    euclid_row = [euclidean_dist(train[j], desc) for j in range(3)]
    cos_row = [cosine_dist(train[j], desc) for j in range(3)]
    kl_row = [kl_divergence(train[j], desc) for j in range(3)]

    euclid_dist.append(euclid_row)
    cos_dist.append(cos_row)
    kl_dist.append(kl_row)

# -------------------------------------------------------
# Print results
# -------------------------------------------------------
print("\n=== Shape Descriptors (Contour-based) ===")
print(tabulate(descriptors,
               headers=["Compactness", "Form Factor", "Roundness", "Eccentricity"],
               showindex=image_name,
               tablefmt="grid"))

# Distance Tables
row_headers = image_name[3:]
col_headers = image_name[:3]

print("\n=== Euclidean Distance Matrix ===")
print(tabulate(euclid_dist, headers=col_headers, showindex=row_headers, tablefmt='grid'))

print("\n=== Cosine Distance Matrix ===")
print(tabulate(cos_dist, headers=col_headers, showindex=row_headers, tablefmt='grid'))

print("\n=== KL Divergence Matrix ===")
print(tabulate(kl_dist, headers=col_headers, showindex=row_headers, tablefmt='grid'))
