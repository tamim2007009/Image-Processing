import numpy as np
import cv2

# ----------------------------
# Function 1: Gaussian smoothing kernel
# ----------------------------
def gaussian_kernel(sigma):
    ksize = int(5 * sigma)  # kernel size proportional to sigma
    if ksize % 2 == 0:
        ksize += 1
    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)
    
    # Correct Gaussian formula
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

    # Laplacian of Gaussian (2nd order derivative)
    kernel = ((xx**2 + yy**2 - 2 * sigma**2) / (sigma**4)) * \
             np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    kernel = kernel - np.mean(kernel)   # normalize
    return kernel.astype(np.float32)

# ----------------------------
# Main
# ----------------------------
# Load color image
img = cv2.imread('lena.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Place 'lena.jpg' in the same folder.")
h, w, c = img.shape

# Take sigma input from user
sigma = float(input("Enter sigma value (e.g., 1.0, 2.0): "))

# Create kernels
smooth_kernel = gaussian_kernel(sigma)
sharp_kernel  = gaussian_derivative_kernel(sigma)

# ----------------------------
# Split channels
# ----------------------------
B, G, R = cv2.split(img)

# For visualization: colorized channels
zeros = np.zeros_like(B)
img_R = cv2.merge([zeros, zeros, R])
img_G = cv2.merge([zeros, G, zeros])
img_B = cv2.merge([B, zeros, zeros])

# ----------------------------
# Apply kernels
# ----------------------------
def process_channel(ch):
    smoothed = cv2.filter2D(ch, -1, smooth_kernel)
    sharpened = cv2.filter2D(ch, -1, sharp_kernel)
    return smoothed, sharpened

B_smooth, B_sharp = process_channel(B)
G_smooth, G_sharp = process_channel(G)
R_smooth, R_sharp = process_channel(R)

# Merge processed channels
img_smooth = cv2.merge([B_smooth, G_smooth, R_smooth])
img_sharp  = cv2.merge([B_sharp,  G_sharp,  R_sharp])

# ----------------------------
# Show results
# ----------------------------
cv2.imshow("Original", img)

cv2.imshow("Red_Channel", img_R)
cv2.imshow("Green_Channel", img_G)
cv2.imshow("Blue_Channel", img_B)

cv2.imshow("Red_Smooth", R_smooth)
cv2.imshow("Red_Sharp", R_sharp)
cv2.imshow("Green_Smooth", G_smooth)
cv2.imshow("Green_Sharp", G_sharp)
cv2.imshow("Blue_Smooth", B_smooth)
cv2.imshow("Blue_Sharp", B_sharp)

cv2.imshow("Smoothed_Image", img_smooth)
cv2.imshow("Sharpened_Image", img_sharp)

# Save all results
cv2.imwrite("Red_Channel.jpg", img_R)
cv2.imwrite("Green_Channel.jpg", img_G)
cv2.imwrite("Blue_Channel.jpg", img_B)

cv2.imwrite("Red_Smooth.jpg", R_smooth)
cv2.imwrite("Red_Sharp.jpg", R_sharp)
cv2.imwrite("Green_Smooth.jpg", G_smooth)
cv2.imwrite("Green_Sharp.jpg", G_sharp)
cv2.imwrite("Blue_Smooth.jpg", B_smooth)
cv2.imwrite("Blue_Sharp.jpg", B_sharp)

cv2.imwrite("Smoothed_Image.jpg", img_smooth)
cv2.imwrite("Sharpened_Image.jpg", img_sharp)

cv2.waitKey(0)
cv2.destroyAllWindows()
