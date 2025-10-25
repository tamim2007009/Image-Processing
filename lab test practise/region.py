# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:45:21 2025

@author: USER
"""

import cv2
import numpy as np
from math import pi, sqrt
from tabulate import tabulate

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

def find_ab(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Fixed: unpack contours properly
    if not contours:  # Added: check if contours are found
        return 0, 0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) >= 5:
        (x,y),(MA,ma),angel = cv2.fitEllipse(cnt)
    else:
        MA, ma = 0, 0
    a = max(MA, ma) / 2.0  # Fixed: ellipse returns diameters, convert to semi-axes
    b = min(MA, ma) / 2.0

    return a, b

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
    
    a, b = find_ab(binary_image)
    eccentricity = np.sqrt(1 - (b/a)**2) if a > 0 else 0  # Fixed: added check for division by zero

    return (compact, form_fact, roundness, eccentricity)  # Fixed: return all 4 descriptors

def find_dist(t1, t2):
    return sqrt((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2 + (t1[3]-t2[3])**2)

def cosine_similarity(t1, t2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(t1, t2))
    magnitude1 = sqrt(sum(a * a for a in t1))
    magnitude2 = sqrt(sum(b * b for b in t2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def kl_distance(t1, t2, epsilon=1e-10):
    """Calculate KL divergence between two distributions"""
    # Convert to probability distributions by adding epsilon to avoid log(0)
    p = np.array(t1) + epsilon
    q = np.array(t2) + epsilon
    
    # Normalize to make probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL divergence: sum(p * log(p/q))
    kl_div = np.sum(p * np.log(p / q))
    return kl_div

# -------------------------------
# Main Code
# -------------------------------

image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png']

descriptors = []
for name in image_name:
    img_gray = cv2.imread(name, 0)
    if img_gray is None:
        print(f"Warning: Could not load image {name}")
        continue

    descriptors.append(calculate_descriptors(img_gray))

# Training descriptors (first 3)
train = descriptors[:3]

# Distance matrix for test images
dist = []
cosine_sim = []  # For cosine similarity
kl_dist = []     # For KL distance

for i in range(3, len(image_name)):
    desc = descriptors[i]
    my_d = [find_dist(train[j], desc) for j in range(3)]
    my_cosine = [cosine_similarity(train[j], desc) for j in range(3)]
    my_kl = [kl_distance(train[j], desc) for j in range(3)]
    
    dist.append(my_d)
    cosine_sim.append(my_cosine)
    kl_dist.append(my_kl)

# Print descriptors table
print("\n=== Shape Descriptors ===")
print(tabulate(descriptors,
               headers=["Compactness","Form Factor","Roundness","Eccentricity"],
               showindex=image_name,
               tablefmt="grid",
               floatfmt=".4f"))

# Print Euclidean distance matrix
print("\n=== Euclidean Distance Matrix ===")
row_headers = image_name[3:]
col_headers = image_name[:3]
print(tabulate(dist, headers=col_headers, showindex=row_headers, tablefmt='grid', floatfmt=".4f"))

# Print Cosine Similarity matrix
print("\n=== Cosine Similarity Matrix ===")
print(tabulate(cosine_sim, headers=col_headers, showindex=row_headers, tablefmt='grid', floatfmt=".4f"))

# Print KL Distance matrix
print("\n=== KL Distance Matrix ===")
print(tabulate(kl_dist, headers=col_headers, showindex=row_headers, tablefmt='grid', floatfmt=".4f"))