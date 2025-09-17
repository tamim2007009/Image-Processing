import cv2
import numpy as np
from math import pi, sqrt
from tabulate import tabulate

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

    return (compact, form_fact, roundness)

def find_dist(t1, t2):
    return sqrt((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2)

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

# Distance matrix for test images
dist = []
for i in range(3, len(image_name)):
    desc = descriptors[i]
    my_d = [find_dist(train[j], desc) for j in range(3)]
    dist.append(my_d)

# Print descriptors table
print("\n=== Shape Descriptors ===")
print(tabulate(descriptors,
               headers=["Compactness","Form Factor","Roundness"],
               showindex=image_name,
               tablefmt="grid"))

# Print distance matrix
print("\n=== Distance Matrix ===")
row_headers = image_name[3:]
col_headers = image_name[:3]
print(tabulate(dist, headers=col_headers, showindex=row_headers, tablefmt='grid'))
