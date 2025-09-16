# -*- coding: utf-8 -*-
"""
Histogram Matching with Erlang Target Distribution
Created on Tue Sep 16 22:58:01 2025

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math



def calc_hist(image):
    h, w = image.shape

    hist = np.zeros(256, dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            p_val = image[i, j]
            hist[p_val] += 1
    return hist   

def calc_pdf(image):
    h, w = image.shape
    total = h * w
    pdf = np.zeros(256, dtype=np.float32)
    hist=calc_hist(image)
    for i in range(256):
        pdf[i] = hist[i] / float(total)   
    return pdf    


def calc_cdf(image):
    
    pdf=calc_pdf(image)
    cdf = np.zeros(256, dtype=np.float32)
    p = 0
    for i in range(256):
        p += pdf[i]
        cdf[i] = p
    return cdf    



def erlang_pdf(x, k, mu):
    x = np.maximum(x, 0.01)  # avoid zero
    return (x**(k-1) * np.exp(-x/mu)) / (mu**k * math.factorial(k-1))

# Histogram function
def erlang_histogram(k, mu, L=256):
    hist = np.zeros(L, dtype=np.float32)

    # Scale x to capture the peak (approx 2*mean)
    max_x = max(20, 2 * k * mu)

    for i in range(L):
        x_val = i * max_x / 255.0
        hist[i] = erlang_pdf(x_val, k, mu)

    # Normalize histogram to sum = 1
    hist = hist / hist.sum()

    return hist



def histogram_matching(image, erlang_hist):

    input_cdf = calc_cdf(image)

  
    target_pdf = erlang_hist / np.sum(erlang_hist)
    target_cdf = np.cumsum(target_pdf)

    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(input_cdf[i] - target_cdf)
        mapping[i] = np.argmin(diff)


    h, w = image.shape
    matched = np.zeros_like(image, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            matched[i, j] = mapping[image[i, j]]

    return matched

#main

# Load grayscale image
image = cv2.imread("histogram.jpg", cv2.IMREAD_GRAYSCALE)

k = int(input("Enter shape parameter k (positive integer): "))
mu = float(input("Enter scale parameter μ (positive real number): "))
target_hist = erlang_histogram(k, mu)

# Apply histogram matching
matched_image = histogram_matching(image, target_hist)

# Display results
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(matched_image, cmap='gray')
plt.title("Histogram Matched Image")
plt.show()
# -------------------------------
# Plot Histogram, PDF, and CDF
# -------------------------------
# Compute hist, pdf, cdf
hist_in = calc_hist(image)
pdf_in = calc_pdf(image)
cdf_in = calc_cdf(image)

hist_out = calc_hist(matched_image)
pdf_out = calc_pdf(matched_image)
cdf_out = calc_cdf(matched_image)

# Plotting in 3x2 grid
plt.figure(figsize=(12, 10))

# Row 1: Histograms
plt.subplot(3,2,1)
plt.plot(hist_in, color='blue')
plt.title('Input Histogram')

plt.subplot(3,2,2)
plt.plot(hist_out, color='red')
plt.title('Matched Histogram')

# Row 2: PDFs
plt.subplot(3,2,3)
plt.plot(pdf_in, color='blue')
plt.title('Input PDF')

plt.subplot(3,2,4)
plt.plot(pdf_out, color='red')
plt.title('Matched PDF')

# Row 3: CDFs
plt.subplot(3,2,5)
plt.plot(cdf_in, color='blue')
plt.title('Input CDF')

plt.subplot(3,2,6)
plt.plot(cdf_out, color='red')
plt.title('Matched CDF')

plt.tight_layout()
plt.show()

