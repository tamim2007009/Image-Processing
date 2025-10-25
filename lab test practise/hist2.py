# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 22:09:11 2025
@author: USER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    h, w = image.shape
    total = h * w

    # Step 1: Compute Histogram
    hist = np.zeros(256, dtype=float)
    for i in range(h):
        for j in range(w):
            p_val = image[i, j]
            hist[p_val] += 1
    plt.plot(hist)
    plt.title('Histogram')
    plt.show()

    # Step 2: Compute PDF
    pdf = hist / total
    plt.plot(pdf)
    plt.title('PDF')
    plt.show()

    # Step 3: Compute CDF
    cdf = np.zeros(256, dtype=float)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    plt.plot(cdf)
    plt.title('CDF')
    plt.show()

    # Step 4: Transformation Function
    trans = np.round(cdf * 255).astype(np.uint8)

    # Step 5: Create Output Image
    out_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out_image[i, j] = trans[image[i, j]]

    # Optional: Output Histogram
    out_hist = np.zeros(256, dtype=float)
    for i in range(h):
        for j in range(w):
            p = out_image[i, j]
            out_hist[p] += 1    

    plt.plot(out_hist)
    plt.title("Equalized Histogram")
    plt.show()

    return out_image

# ===== Main Program =====
img = cv2.imread('image.jpg', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)
v_out = histogram_equalization(v)

output_image = cv2.merge([h, s, v_out])

# Convert HSV back to BGR for display
output_bgr = cv2.cvtColor(output_image, cv2.COLOR_HSV2BGR)

cv2.imshow("Input Image", img)
cv2.waitKey(0)

cv2.imshow("Output Image (Equalized)", output_bgr)
cv2.waitKey(0)

cv2.destroyAllWindows()
