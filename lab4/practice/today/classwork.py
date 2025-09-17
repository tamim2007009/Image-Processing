import cv2
import numpy as np
from math import pi, sqrt
from tabulate import tabulate

# -------------------------------
# Functions
# -------------------------------

def find_max_d(binary_image):
    min_x = min_y = 100000
    max_x = max_y = 0
    
    h, w = binary_image.shape
    for x in range(h):
        for y in range(w):
            if binary_image[x, y] <= 0:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    return max(max_x - min_x, max_y - min_y)

def calculate_descriptors(binary_image, i):
    # Erode to find border
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    border_image = binary_image - eroded
    
    area = np.count_nonzero(binary_image)
    perimeter = np.count_nonzero(border_image)
    max_d = find_max_d(binary_image)
    
    # Show images
    cv2.imshow(f'Border {i}', border_image)   
    
    cv2.imshow(f'Input Image {i}', binary_image)
    
    # Shape descriptors
    compact = (perimeter**2) / area
    form_fact = (4 * pi * area) / (perimeter**2) 
    roundness = (4 * area) / (pi * max_d**2)
    
    return (compact, form_fact, roundness)

def find_dist(t1, t2):
    return sqrt((t1[0] - t2[0])**2 + (t1[1] - t2[1])**2 + (t1[2] - t2[2])**2)

def show(distances_matrix):
    row_headers = ['c2.jpg','t2.jpg','p2.png','st.jpg']
    col_headers = ['c1.jpg','t1.jpg','p1.png']
    distances_matrix = np.array(distances_matrix)
    print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))

def write_to_file(im_title, main_data):
    file_path = 'output2.txt'  # saved in current folder
    with open(file_path, 'w') as file:
        file.write('\t'.join(['Image', 'Form Factor', 'Roundness', 'Compactness']) + '\n')
        file.write('-' * 50 + '\n')
        for i, row in enumerate(main_data):
            file.write(im_title[i] + '\t\t')
            line = '\t\t'.join(map(str, row))
            file.write(line + '\n')
            file.write('-' * 50 + '\n')

# -------------------------------
# Main Code
# -------------------------------

image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png','st.jpg']

# Calculate descriptors for training images
train = []
for i in range(3):
    img = cv2.imread(image_name[i], 0)  # grayscale
    if img is None:
        raise FileNotFoundError(f"Image {image_name[i]} not found!")
    train.append(calculate_descriptors(img, i))

# Calculate descriptors for test images and distances
test = []
dist = []
for i in range(3, len(image_name)):
    img = cv2.imread(image_name[i], 0)
    if img is None:
        raise FileNotFoundError(f"Image {image_name[i]} not found!")
    
    c_f_r = calculate_descriptors(img, i)
    test.append(c_f_r)
    
    my_d = []
    for j in range(3):
        my_d.append(find_dist(train[j], c_f_r))
    dist.append(my_d)

# Append test descriptors to train
train.extend(test)

# Write descriptors to file
write_to_file(image_name, train)

# Display distance matrix
show(dist)

cv2.waitKey(0)
cv2.destroyAllWindows()
