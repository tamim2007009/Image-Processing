import cv2
import numpy as np
from math import pi
from math import sqrt
from tabulate import tabulate
import random

def find_max_d(binary_image):
    min_x = min_y = 100000
    max_x = max_y = 0
    
    h,w = binary_image.shape
    
    for x in range(h):
        for y in range(w):
            if(binary_image[x,y] <= 0):
                continue
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    return max( max_x - min_x, max_y - min_y )

def calculate_descriptors(binary_image,i):
    #_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    border_image = binary_image - eroded
    
    area = np.count_nonzero(binary_image)
    perimeter = np.count_nonzero(border_image)
    
    max_d = find_max_d(binary_image)
    
    cv2.imshow('Border'+str(i), border_image)   
    cv2.imshow('Input image'+str(i), img)
    
    compact = (perimeter**2) / area
    form_fact = (4*pi*area) / (perimeter**2)
    roundness = (4*area) / (pi * max_d**2)
    
    return (compact, form_fact, roundness)

def find_dist(t1, t2):
    abs_c = (t1[0] - t2[0])**2
    abs_f = (t1[1] - t2[1])**2
    abs_r = (t1[2] - t2[2])**2
    
    return sqrt( abs_c + abs_f + abs_r )


def show(distances_matrix):
    row_headers = ['c2.jpg','t2.jpg','p2.png', 'st.jpg']
    col_headers = ['c1.jpg','t1.jpg','p1.png']

    distances_matrix = np.array(distances_matrix)
    # Display the distance matrix as a table
    print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))

def write_to_file(im_title, main_data):
    file_path = 'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\ClassWork_5\\output2.txt'
    with open(file_path, 'w') as file:
        # Write the column names
        file.write('\t'.join(map(str, [' ', 'form_factor', 'roundness','compactness'])) + '\n')
        # Write horizontal line
        file.write('-' * 50 + '\n')
        i=0
        for row in main_data:
            file.write(im_title[i]+'\t\t')
            line = '\t\t'.join(map(str, row ))
            i=i+1
            # Write the line to the file
            file.write(line + '\n')
            # Write horizontal line
            file.write('-' * 50 + '\n')


image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png', 'st.jpg']

train = []
for i in range(3):
    root = 'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\ClassWork_5\\'
    image_path = root+image_name[i]
    
    img = cv2.imread(image_path, 0)
    train.append( calculate_descriptors(img,i) )

print(train)

test = []
dist = []
for i in range(3,len(image_name)):
    root = 'D:\\Documents\\COURSES\\4.1\\Labs\\Image\\ImageCodes\\ClassWork_5\\'
    image_path = root+image_name[i]
    
    img = cv2.imread(image_path, 0)
    
    c_f_r = calculate_descriptors(img,i)
    test.append( c_f_r )
    
    my_d = []
    for j in range(3):
        my_d.append( find_dist(train[j], c_f_r) )    
    dist.append(my_d)

# print(train)
# print(test)
print(dist)

for item in test:
    train.append(item)

write_to_file(image_name, train)
show(dist)

cv2.waitKey(0)
cv2.destroyAllWindows()
