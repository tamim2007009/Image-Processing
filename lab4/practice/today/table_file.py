import cv2
import numpy as np
import math
from tabulate import tabulate
import random

#Table Creation


#File Creation
im_title = ['c1','t1','s1','c2','t2','s2','st']
file_path = 'output2.txt'
with open(file_path, 'w') as file:
    # Write the column names
    file.write('\t'.join(map(str, [' ', 'form_factor', 'roundness','compactness'])) + '\n')
    # Write horizontal line
    file.write('-' * 50 + '\n')
    i=0
    for row in distances_matrix:
        file.write(im_title[i]+'\t\t')
        line = '\t\t'.join(map(str, row ))
        i=i+1
        # Write the line to the file
        file.write(line + '\n')
        # Write horizontal line
        file.write('-' * 50 + '\n')
cv2.waitKey(0)
cv2.destroyAllWindows()