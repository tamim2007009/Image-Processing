   # -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 17:28:21 2025

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def histogram(image):
    h, w = image.shape
    total = h * w
    hist = np.zeros(256, dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            p_val = image[i, j]
            hist[p_val] += 1

    pdf = np.zeros(256, dtype=np.float32)
    for i in range(256):
        pdf[i] = hist[i] / float(total)
 
    cdf = np.zeros(256, dtype=np.float32)
    p = 0
    for i in range(256):
        p += pdf[i]
        cdf[i] = p

    trans = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        trans[i] = int(round(255 * cdf[i]))

    out_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out_image[i, j] = trans[image[i, j]]

    return out_image

img = cv2.imread("image.jpg", 1)

b, g, r = cv2.split(img)
b_out = histogram(b)
g_out = histogram(g)
r_out = histogram(r)
merged = cv2.merge([b_out, g_out, r_out])

cv2.imshow("original image", img)
cv2.waitKey(0)

cv2.imshow("blue channel", b_out)
cv2.waitKey(0)

cv2.imshow("green channel", g_out)
cv2.waitKey(0)

cv2.imshow("red channel", r_out)
cv2.waitKey(0)

cv2.imshow("merge channel", merged)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v_out = histogram(v)
hsv_merge = cv2.merge([h, s, v_out])
final_rgb = cv2.cvtColor(hsv_merge, cv2.COLOR_HSV2BGR)


cv2.imshow("Hue  channel", h)
cv2.waitKey(0)

cv2.imshow("saturation channel", s)
cv2.waitKey(0)
cv2.imshow("value channel", v)
cv2.waitKey(0)


cv2.imshow("value channel equalized", v_out)
cv2.waitKey(0)

cv2.imshow("combined HSV equalized", final_rgb)
cv2.waitKey(0)

cv2.destroyAllWindows()
