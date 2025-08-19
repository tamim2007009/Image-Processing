import numpy as np
import cv2

def gaussian_kernel(sigma):
    
    size = int(5 * sigma)         
    if size % 2 == 0:
        size += 1               

    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    xr, yc = np.meshgrid(x, y)   

    kernel = np.exp(-(xr**2 + yc**2) / (2 * sigma**2))


    kernel = kernel / np.sum(kernel)

    return kernel


def sharpening_kernel(sigma):
  
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1

    
    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    xr, yc = np.meshgrid(x, y)
    
    kernel = ((xr**2 + yc**2 - 2*sigma**2) / (sigma**4)) * np.exp(-(xr**2 + yc**2) / (2*sigma**2))

    kernel = kernel - np.mean(kernel)

    return kernel



img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)


sigma = float(input("Enter sigma value: "))

cv2.imshow("Original", img)
cv2.waitKey(0)

smooth_kernel = gaussian_kernel(sigma)
img_smooth = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=smooth_kernel)
smooth_norm = np.round(cv2.normalize(img_smooth, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow("Gaussian Smoothed image", smooth_norm)
cv2.waitKey(0)

sharp_kernel = sharpening_kernel(sigma)
img_sharp = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=sharp_kernel)
sharp_norm = np.round(cv2.normalize(img_sharp, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow("Gaussian Sharpened image", sharp_norm)
cv2.waitKey(0)


cv2.destroyAllWindows()
