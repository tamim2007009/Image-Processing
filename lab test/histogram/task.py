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

# First equalization
he1, hist, out_hist, cdf, out_cdf = Manual_equalizeHist(image)

# Second equalization on HE1
he2, hist2, out_hist2, cdf2, out_cdf2 = Manual_equalizeHist(he1)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(14, 12))

# 1. Input Image
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Input Image")
plt.axis("off")

# 2. HE1 Image
plt.subplot(3, 3, 2)
plt.imshow(he1, cmap='gray')
plt.title("HE1 Image")
plt.axis("off")

# 3. HE2 Image
plt.subplot(3, 3, 3)
plt.imshow(he2, cmap='gray')
plt.title("HE2 Image")
plt.axis("off")

# 4. Input Histogram
plt.subplot(3, 3, 4)
plt.plot(hist, color='blue')
plt.title("Input Histogram")

# 5. Histogram of HE1
plt.subplot(3, 3, 5)
plt.plot(out_hist, color='green')
plt.title("Histogram of HE1")

# 6. Histogram of HE2
plt.subplot(3, 3, 6)
plt.plot(out_hist2, color='red')
plt.title("Histogram of HE2")

# 7. Input CDF
plt.subplot(3, 3, 7)
plt.plot(cdf, color='blue')
plt.title("Input CDF")

# 8. CDF of HE1
plt.subplot(3, 3, 8)
plt.plot(out_cdf, color='green')
plt.title("CDF of HE1")

# 9. CDF of HE2
plt.subplot(3, 3, 9)
plt.plot(out_cdf2, color='red')
plt.title("CDF of HE2")

plt.tight_layout()
plt.show()


# -----------------------------
# Extra: Compare HE1 and HE2
# -----------------------------
diff = cv2.absdiff(he1, he2)
mse = np.mean((he1.astype("float") - he2.astype("float")) ** 2)
psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

print("MSE between HE1 and HE2:", mse)
print("PSNR between HE1 and HE2:", psnr)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(diff, cmap='gray')
plt.title("Difference (HE1 - HE2)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.plot(out_hist, label="HE1 Histogram", color="green")
plt.plot(out_hist2, label="HE2 Histogram", color="red", linestyle="--")
plt.legend()
plt.title("Histogram Comparison: HE1 vs HE2")
plt.show()
