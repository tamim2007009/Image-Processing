import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Method 1: Equalize Each RGB Channel Separately ---
def equalize_rgb(image):
    # Split into channels
    b, g, r = cv2.split(image)

    # Equalize each channel separately
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # Merge back
    eq_img = cv2.merge([b_eq, g_eq, r_eq])
    return eq_img


# --- Method 2: Equalize Only the V channel in HSV ---
def equalize_hsv(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Equalize only Value channel
    v_eq = cv2.equalizeHist(v)

    # Merge back with original H and S
    hsv_eq = cv2.merge([h, s, v_eq])

    # Convert back to BGR for display
    eq_img = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return eq_img


# -------- Main Program --------
image = cv2.imread(r"lena.jpg",1)

# Method 1: RGB equalization
rgb_eq = equalize_rgb(image)

# Method 2: HSV equalization
hsv_eq = equalize_hsv(image)

# Show results
cv2.imshow("Original", image)
cv2.imshow("RGB Equalized", rgb_eq)
cv2.imshow("HSV Equalized (Value channel)", hsv_eq)

cv2.waitKey(0)
cv2.destroyAllWindows()
