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
    kernel = kernel - np.mean(kernel)   # normalization
    return kernel


img = cv2.imread('lena.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(img_hsv)



sigma = float(input("Enter sigma value: "))
smooth_kernel = gaussian_kernel(sigma)
sharp_kernel  = sharpening_kernel(sigma)


H_smooth = cv2.filter2D(H, -1, smooth_kernel)
S_smooth = cv2.filter2D(S, -1, smooth_kernel)
V_smooth = cv2.filter2D(V, -1, smooth_kernel)

hsv_smooth = cv2.merge([H_smooth, S_smooth, V_smooth])




H_sharp = cv2.filter2D(H, -1, sharp_kernel)
S_sharp = cv2.filter2D(S, -1, sharp_kernel)
V_sharp = cv2.filter2D(V, -1, sharp_kernel)
hsv_sharp = cv2.merge([H_sharp, S_sharp, V_sharp])



cv2.imshow("Original (BGR)", img); cv2.waitKey(0)

cv2.imshow("Hue Channel", H); cv2.waitKey(0)
cv2.imshow("Saturation Channel", S); cv2.waitKey(0)
cv2.imshow("Value Channel", V); cv2.waitKey(0)

cv2.imshow("Gausian Hue", H_smooth); cv2.waitKey(0)
cv2.imshow("Gausian Saturation", S_smooth); cv2.waitKey(0)
cv2.imshow("Gausian Value", V_smooth); cv2.waitKey(0)
cv2.imshow("Gausian merged", hsv_smooth); cv2.waitKey(0)

cv2.imshow("Sharpened Hue", H_sharp); cv2.waitKey(0)
cv2.imshow("Sharpened Saturation", S_sharp); cv2.waitKey(0)
cv2.imshow("Sharpened Value", V_sharp); cv2.waitKey(0)
cv2.imshow("Sharpened merged", hsv_sharp); cv2.waitKey(0)

cv2.destroyAllWindows()
