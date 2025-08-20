

import numpy as np
import cv2

# Load grayscale image
img = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
print(h)
print(w)


img_bordered = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
cv2.imshow('Original Grayscale Image', img)
cv2.waitKey(0)
# Gaussian kernel
kernel1 = np.array([
    [0,  1,  2,  1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 9, 5, 2],
    [1, 3, 5, 3, 1],
    [0,  1, 2,  1, 0]
], dtype=np.float32)

kernel1=np.flip(kernel1)

out_img = np.zeros((h, w), dtype=np.float32)

 
for i in range(h):
    for j in range(w):
        region=img_bordered[i:i+5,j:j+5]
        result=np.sum(region*kernel1)
        out_img[i,j]=result

# Normalize the float result to 0–255 and convert to uint8
norm = np.round(cv2.normalize(out_img, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)


cv2.imshow('Bordered Image', img_bordered)
cv2.waitKey(0)

cv2.imshow('Normalized Image', norm)
cv2.waitKey(0)
cv2.destroyAllWindows()


#task2
# kernel2 = np.array([[-1, 0, 1],
#                     [-1, 0, 1],
#                     [-1, 0, 1]])

# kernel3 = np.array([[-1, -1, -1],
#                     [0, 0, 0],
#                     [1, 1, 1]])

# img_bordered=cv2.copyMakeBorder(img, 0, 2, 0, 2, cv2.BORDER_CONSTANT)
# kernel2=np.flip(kernel2)

# out_img = np.zeros((h, w), dtype=np.float32)

# for i in range(h):
#     for j in range(w):
#         region=img_bordered[i:i+3,j:j+3]
#         result=np.sum(region*kernel2)
#         out_img[i,j]=result
        
# norm=np.round(cv2.normalize(out_img, None,0,255,cv2.NORM_MINMAX)).astype(np.uint8)        
        
# cv2.imshow('Bordered Image', img_bordered)
# cv2.waitKey(0)

# cv2.imshow('Gx:Normalized Image', norm)
# cv2.waitKey(0)




# out_img2 = np.zeros((h, w), dtype=np.float32)

# for i in range(h):
#     for j in range(w):
#         region=img_bordered[i:i+3,j:j+3]
#         result=np.sum(region*kernel3)
#         out_img2[i,j]=result
        
# norm2=np.round(cv2.normalize(out_img2, None,0,255,cv2.NORM_MINMAX)).astype(np.uint8)        
        


# cv2.imshow('Gy:Normalized Image', norm)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

 

 

