# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 00:05:03 2025

@author: USER
"""

import cv2
import numpy as np

# 1️⃣ Generate Laplacian of Gaussian (LoG) filter
def generate_log_filter(sigma):
    size = int(9 * sigma)
    if size % 2 == 0:
        size += 1  # kernel must be odd
    center = size // 2

    x = np.arange(-center, center + 1)
    y = np.arange(-center, center + 1)
    X, Y = np.meshgrid(x, y)

    # LoG equation
    log_filter = (-1 / (np.pi * sigma**4)) * (
        1 - ((X**2 + Y**2) / (2 * sigma**2))
    ) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    return log_filter


# 2️⃣ Apply LoG filter on image
def apply_log(image, log_filter):
    # convert to float and apply convolution
    return cv2.filter2D(image.astype(np.float32), -1, log_filter)


# 3️⃣ Detect zero-crossings (4-neighborhood)
def zero_crossing_detection(log_response, threshold):
    h, w = log_response.shape
    output = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            current = log_response[i, j]
            
            # Check if sign changes with any neighbor
            if (current * log_response[i-1, j] < 0) or \
               (current * log_response[i+1, j] < 0) or \
               (current * log_response[i, j-1] < 0) or \
               (current * log_response[i, j+1] < 0):
                
                # Simple strength calculation
                strength = abs(current - log_response[i-1, j]) + \
                          abs(current - log_response[i+1, j]) + \
                          abs(current - log_response[i, j-1]) + \
                          abs(current - log_response[i, j+1])
                
                if strength > threshold:
                    output[i, j] = 255
                    
    return output


# 4️⃣ MAIN CODE
def main():
    # Load grayscale image
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    

    sigma = float(input("Enter sigma value (e.g., 1.0 or 2.0): "))
    threshold = float(input("Enter Zero-cross strength threshold (e.g., 5): "))

    # Generate LoG filter
    log_filter = generate_log_filter(sigma)
    print(f"✅ LoG filter generated with size {log_filter.shape}")

    # Apply convolution
    log_response = apply_log(img, log_filter)

    # Normalize for display
    log_display = cv2.normalize(log_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Zero-cross detection
    zero_cross_img = zero_crossing_detection(log_response, threshold)

    # Show results
    cv2.imshow("Original Image", img)
    cv2.imshow("LoG Response", log_display)
    cv2.imshow("Zero Crossing Edge", zero_cross_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
