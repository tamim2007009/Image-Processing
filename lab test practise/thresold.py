# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 00:01:20 2025

@author: USER
"""

import numpy as np
import cv2

def gaussian_derivative_kernels(sigma):
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1
    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    X, Y = np.meshgrid(x, y)
    const = 1.0 / (2.0 * np.pi * (sigma ** 2))
    gaussian = const * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    Gx = -(X / (sigma**2)) * gaussian
    Gy = -(Y / (sigma**2)) * gaussian
    return Gx, Gy

def threshold_image(img, th1,th2):
    final= np.zeros_like(img)
    h,w=img.shape
    for i in range(h):
        for j in range(w):
            if img[i,j]>th1 and img[i,j]<th2:
                final[i,j]=128
            elif img[i,j]>th2:
                final[i,j]=255
            else:
                final[i,j]=0
    return final

def normalize_kernel(K):
    K_norm = cv2.normalize(K, None, alpha=-25, beta=25, norm_type=cv2.NORM_MINMAX)
    return np.round(K_norm).astype(int) 

sigma = float(input("Enter the value of sigma: "))
gx, gy = gaussian_derivative_kernels(sigma)

print("x derivative kernel after normalization\n", normalize_kernel(gx))
print("y derivative kernel after normalization\n", normalize_kernel(gy))





img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

xcon = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=gx)
ycon = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=gy)

xnorm = np.round(cv2.normalize(xcon, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
ynorm = np.round(cv2.normalize(ycon, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.imshow("x derivative image", xnorm)
cv2.waitKey(0)
cv2.imshow("y derivative image", ynorm)
cv2.waitKey(0)

grad_mag = np.sqrt(xcon**2 + ycon**2)
grad_mag = np.round(cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow("gradient image", grad_mag)
cv2.waitKey(0)

th1, th2 = map(int, input("Enter two threshold values: ").split())
th_img = threshold_image(grad_mag, th1, th2)

cv2.imshow("After applying thresholding", th_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
