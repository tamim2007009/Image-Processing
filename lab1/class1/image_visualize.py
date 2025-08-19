import numpy as np
import cv2

img2 = np.array([
    [   12,  26,  33,  26,  12],
    [   26,  55,  71,  55,  26],
    [   33,  71,  91,  71,  33],
    [   26,  55,  71,  55,  26],
    [   12,  26,  33,  26,  12],
], dtype=np.uint8)

# Use a reasonable scaling factor like 100 instead of 500
img3 = cv2.resize(img2, None, fx = 100, fy = 100, interpolation = cv2.INTER_NEAREST)

# Display the resized image (it will look dark, but it's correct)
cv2.imshow('Resized Image (Dark)', img3)
cv2.waitKey(0)

# The multiplication line is removed as it causes overflow
# norm = np.round(cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# Directly normalize the resized image
norm = cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('Correct Normalized Image', norm)
cv2.waitKey(0)
cv2.destroyAllWindows()