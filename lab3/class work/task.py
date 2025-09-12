# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 15:51:07 2025

@author: USER
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Manual_equalizeHist(image):
    h, w = image.shape
    total = h * w
    hist = np.zeros(256, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            pixel_val = image[i, j]
            hist[pixel_val] += 1 

    # plt.figure()
    # plt.plot(hist)
    # plt.title("Histogram")
    # plt.show()

    pdf = np.zeros(256, dtype=np.float32)
    for i in range(256):
        pdf[i] = hist[i] / float(total)
   

    # plt.figure()
    # plt.plot(pdf)
    # plt.title("PDF (Probability Density Function)")
    # plt.show()

   
    cdf = np.zeros(256, dtype=np.float32)
    cumulative = 0.0
    for i in range(256):
        cumulative += pdf[i]
        cdf[i] = cumulative
  
    # plt.figure()
    # plt.plot(cdf)
    # plt.title("CDF (Cumulative Distribution Function)")
    # plt.show()

    trans = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        trans[i] = int(round(cdf[i] * 255))
  

    out = np.zeros_like(image, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = trans[image[i, j]]
    
    out_hist=np.zeros(256,dtype=np.uint32)
    for i in range(h):
        for j in range(w):
            pix_val=out[i,j]
            out_hist[pix_val]+=1
            
    # plt.figure()
    # plt.plot(out_hist)
    # plt.title("output histogram")
    # plt.show()        
            
    out_pdf=np.zeros(256,dtype=np.float32)
    for i in range(256):
        out_pdf[i]=out_hist[i]/float(total)
        
    # plt.figure()
    # plt.plot(out_pdf)
    # plt.title("output pdf")
    # plt.show()      
    

    out_cdf= np.zeros(256,dtype=np.float32)    
    x=0.0
    for i in range(256):
        x+=out_pdf[i]
        out_cdf[i]=x

    # plt.figure()
    # plt.plot(out_cdf)
    # plt.title("output cdf")
    # plt.show()     
    
    
    
    return out, hist, out_hist,cdf, out_cdf

# -------- Main Program --------
image = cv2.imread(r"histogram.jpg", cv2.IMREAD_GRAYSCALE)
he1, hist, out_hist, cdf, out_cdf = Manual_equalizeHist(image)


he2 = cv2.equalizeHist(image)

plt.figure(figsize=(10, 12))

# 1. Input Image
plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Input Image")


# 2. Output Image (HE1 - Manual)
plt.subplot(3, 2, 2)
plt.imshow(he1, cmap='gray')
plt.title("Output Image (HE1 - Manual)")


# 3. Input Histogram
plt.subplot(3, 2, 3)
plt.plot(hist)
plt.title("Input Histogram")


# 4. Histogram of HE1
plt.subplot(3, 2, 4)
plt.plot(out_hist)
plt.title("Histogram of HE1")


# 5. Input CDF
plt.subplot(3, 2, 5)
plt.plot(cdf)
plt.title("Input CDF")


# 6. CDF of HE1
plt.subplot(3, 2, 6)
plt.plot(out_cdf)
plt.title("CDF of HE1")

cv2.waitKey(0)
cv2.destroyAllWindows()