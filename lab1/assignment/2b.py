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
    
    # Gaussian function
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
    kernel = ((xx**2 + yy**2 - 2 * (sigma**2)) / (sigma**4)) * \
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

# Convert BGR -> HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Split HSV channels
H, S, V = cv2.split(img_hsv)

# Kernels
sigma_smooth = 1.0
smooth_kernel = gaussian_kernel(sigma_smooth)

sigma_sharp = 1.0
sharp_kernel = gaussian_derivative_kernel(sigma_sharp)

# ----------------------------
# Apply convolution to each channel
# ----------------------------
def process_channel(ch):
    smoothed = cv2.filter2D(ch, -1, smooth_kernel)
    sharpened = cv2.filter2D(ch, -1, sharp_kernel)
    return smoothed, sharpened

H_smooth, H_sharp = process_channel(H)
S_smooth, S_sharp = process_channel(S)
V_smooth, V_sharp = process_channel(V)

# Merge results into HSV images
hsv_smooth = cv2.merge([H_smooth, S_smooth, V_smooth])
hsv_sharp  = cv2.merge([H_sharp,  S_sharp,  V_sharp])

# Convert back to BGR for visualization
img_smooth_bgr = cv2.cvtColor(hsv_smooth, cv2.COLOR_HSV2BGR)
img_sharp_bgr  = cv2.cvtColor(hsv_sharp,  cv2.COLOR_HSV2BGR)

# ----------------------------
# Show / Save results
# ----------------------------
cv2.imshow("Original (BGR)", img)
cv2.imshow("Original HSV - H", H)
cv2.imshow("Original HSV - S", S)
cv2.imshow("Original HSV - V", V)

cv2.imshow("H Smooth", H_smooth)
cv2.imshow("H Sharp", H_sharp)

cv2.imshow("S Smooth", S_smooth)
cv2.imshow("S Sharp", S_sharp)

cv2.imshow("V Smooth", V_smooth)
cv2.imshow("V Sharp", V_sharp)

cv2.imshow("Final Smoothed (HSV->BGR)", img_smooth_bgr)
cv2.imshow("Final Sharpened (HSV->BGR)", img_sharp_bgr)

# Save all results
cv2.imwrite("HSV_H.jpg", H)
cv2.imwrite("HSV_S.jpg", S)
cv2.imwrite("HSV_V.jpg", V)

cv2.imwrite("H_Smooth.jpg", H_smooth)
cv2.imwrite("H_Sharp.jpg", H_sharp)

cv2.imwrite("S_Smooth.jpg", S_smooth)
cv2.imwrite("S_Sharp.jpg", S_sharp)

cv2.imwrite("V_Smooth.jpg", V_smooth)
cv2.imwrite("V_Sharp.jpg", V_sharp)

cv2.imwrite("HSV_Smoothed_BGR.jpg", img_smooth_bgr)
cv2.imwrite("HSV_Sharpened_BGR.jpg", img_sharp_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
