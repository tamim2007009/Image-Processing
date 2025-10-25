# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 19:10:29 2025
@author: USER
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt

def histogram_equalization(image):
    h, w = image.shape
    total = h * w

    # ----- Step 1: Compute Histogram -----
    hist = np.zeros(256, dtype=float)
    for i in range(h):
        for j in range(w):
            p_val = image[i, j]
            hist[p_val] += 1
    plt.plot(hist)
    plt.title('Histogram')
    plt.show()
    # ----- Step 2: Compute PDF -----
    pdf = hist / total
    plt.plot(pdf)
    plt.title('pdf')
    plt.show()
    # ----- Step 3: Compute CDF -----
    cdf = np.zeros(256, dtype=float)
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + pdf[i]
    plt.plot(cdf)
    plt.title('cdf')
    plt.show()
    
    # ----- Step 4: Transformation Function -----
    trans = np.round(cdf * 255).astype(np.uint8)

    # ----- Step 5: Create Output Image -----
    out_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out_image[i, j] = trans[image[i, j]]

    
    out_hist=np.zeros(256, dtype=float)
    for i in range(h):
        for j in range(w):
            p = out_image[i, j]
            out_hist[p]+= 1    
            
            
    out_pdf= np.zeros(256, dtype=float)        
    out_pdf=out_hist/total
    out_cdf=np.zeros(256, dtype=float) 
    out_cdf=np.cumsum(out_pdf)
    
    
    plt.plot(out_hist)
    plt.title('Histogram')
    plt.show()
    
    plt.plot(out_pdf)
    plt.title('Histogram')
    plt.show()
    
    plt.plot(out_cdf)
    plt.title('Histogram')
    plt.show()
    

    return out_image


# ----- Main Program -----
img = cv2.imread('image.jpg', 1)

cv2.imshow("Original Image", img)
cv2.waitKey(0)

b,r,g=cv2.split(img)

b_out=histogram_equalization(b)
cv2.imshow("Blue channel",b_out)
cv2.waitKey(0)

r_out=histogram_equalization(r)
cv2.imshow("Red channel", r_out)

cv2.waitKey(0)
g_out=histogram_equalization(g)

cv2.imshow("Green channel", g_out)

cv2.waitKey(0)

merged_image=cv2.merge([b_out,g_out,r_out])

cv2.imshow("Merge channel output", merged_image)




cv2.waitKey(0)
cv2.destroyAllWindows()
