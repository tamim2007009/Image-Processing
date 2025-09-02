import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram(image):
    h, w = image.shape
    total = h * w
    print(f"Image size: {h} x {w}, Total pixels: {total}")
    hist = np.zeros(256, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            pixel_val = image[i, j]
            hist[pixel_val] += 1 

    plt.figure()
    plt.plot(hist)
    plt.title("Histogram")
    plt.show()

    pdf = np.zeros(256, dtype=np.float32)
    for i in range(256):
        pdf[i] = hist[i] / float(total)
   

    plt.figure()
    plt.plot(pdf)
    plt.title("PDF (Probability Density Function)")
    plt.show()

   
    cdf = np.zeros(256, dtype=np.float32)
    cumulative = 0.0
    for i in range(256):
        cumulative += pdf[i]
        cdf[i] = cumulative
  
    plt.figure()
    plt.plot(cdf)
    plt.title("CDF (Cumulative Distribution Function)")
    plt.show()

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
            
    plt.figure()
    plt.plot(out_hist)
    plt.title("output histogram")
    plt.show()        
            
    out_pdf=np.zeros(256,dtype=np.float32)
    for i in range(256):
        out_pdf[i]=out_hist[i]/float(total)
        
    plt.figure()
    plt.plot(out_pdf)
    plt.title("output pdf")
    plt.show()      
    

    out_cdf= np.zeros(256,dtype=np.float32)    
    x=0.0
    for i in range(256):
        x+=out_pdf[i]
        out_cdf[i]=x

    plt.figure()
    plt.plot(out_cdf)
    plt.title("output cdf")
    plt.show()     
    
    
    
    return hist, pdf, cdf, out


# -------- Main Program --------
image = cv2.imread(r"C:\Users\USER\Desktop\image\Image-Processing\lab3\practice\histogram.jpg", cv2.IMREAD_GRAYSCALE)

hist, pdf, cdf, equalized_img = histogram(image)

# Show original vs equalized images
cv2.imshow("Original Image", image)

cv2.imshow("Equalized Image (Manual)", equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
