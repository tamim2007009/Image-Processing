# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 18:52:34 2025

@author: USER
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---------------- Histogram Equalization for grayscale ----------------
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

def show_pdf_cdf(image):
    chans = cv2.split(image)
    pdf_list = []
    cdf_list = []

    for chan in chans:
        h, w = chan.shape
        total = h * w
        hist = np.zeros(256, dtype=np.uint32)

        # Histogram
        for i in range(h):
            for j in range(w):
                p_val = chan[i, j]
                hist[p_val] += 1

        # PDF (manual loop)
        pdf = np.zeros(256, dtype=np.float32)
        for i in range(256):
            pdf[i] = hist[i] / float(total)

        # CDF (manual loop)
        cdf = np.zeros(256, dtype=np.float32)
        cumulative = 0.0
        for i in range(256):
            cumulative += pdf[i]
            cdf[i] = cumulative

        # Save results
        pdf_list.append(pdf)
        cdf_list.append(cdf)

    return pdf_list, cdf_list




# ---------------- Plot Histograms ----------------
def plot_diagrams(hist_list, title):
    colors = ("b", "g", "r")
    for hist, color in zip(hist_list, colors):
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.title(title)


# ---------------- Main Code ----------------
img = cv2.imread("image.jpg", 1)  


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v_out = histogram(v)  
hsv_merge = cv2.merge([h, s, v_out])
final_rgb = cv2.cvtColor(hsv_merge, cv2.COLOR_HSV2BGR)

# Show images (optional)
cv2.imshow("Original Image", img)
cv2.imshow("Equalized Image", final_rgb)
cv2.waitKey(0)

in_pdf,in_cdf = show_pdf_cdf(img)
out_pdf, out_cdf = show_pdf_cdf(final_rgb)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_diagrams(in_pdf, "Input pdf")

plt.subplot(1, 2, 2)
plot_diagrams(in_cdf, "input cdf")

plt.show()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_diagrams(out_pdf, "output pdf")

plt.subplot(1, 2, 2)
plot_diagrams(out_cdf, "output cdf")

plt.show()



cv2.destroyAllWindows()
