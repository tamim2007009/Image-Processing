import numpy as np
import cv2

# ----------------------------
# Function 1: Gaussian smoothing kernel
# ----------------------------
def gaussian_kernel(sigma):
    ksize = int(5 * sigma)  # kernel size proportional to sigma
    if ksize % 2 == 0:   # ensure odd size
        ksize += 1
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    
    # Gaussian formula
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)

# ----------------------------
# Function 2: Gaussian derivative (Laplacian of Gaussian) kernel
# ----------------------------
def gaussian_derivative_kernel(sigma):
    ksize = int(7 * sigma)  # bigger window for derivative
    if ksize % 2 == 0:
        ksize += 1
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)

    # Laplacian of Gaussian (LoG) – 2nd order derivative
    kernel = ((xx**2 + yy**2 - 2 * (sigma**2)) / (sigma**4)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize to keep values balanced
    kernel = kernel - np.mean(kernel)
    return kernel.astype(np.float32)

# ----------------------------
# Main
# ----------------------------
# Load grayscale image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Place 'lena.jpg' in the working directory.")

# Take sigma as user input
sigma = float(input("Enter sigma value (e.g. 1.0, 2.0): "))

# 1. Smoothing
smooth_kernel = gaussian_kernel(sigma)
img_smooth = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=smooth_kernel)

# Normalize
smooth_norm = np.round(cv2.normalize(img_smooth, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# 2. Sharpening
sharp_kernel = gaussian_derivative_kernel(sigma)
img_sharp = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=sharp_kernel)

# Normalize
sharp_norm = np.round(cv2.normalize(img_sharp, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# ----------------------------
# Show all results
# ----------------------------
cv2.imshow("Original", img)
cv2.imshow("Gaussian Smoothed", smooth_norm)
cv2.imshow("Gaussian Derivative (Sharpened)", sharp_norm)

cv2.waitKey(0)
cv2.destroyAllWindows()
