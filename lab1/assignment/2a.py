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




img = cv2.imread('lena.jpg', 1)
B, G, R = cv2.split(img)
sigma = float(input("Enter sigma value: "))


smooth_kernel = gaussian_kernel(sigma)
sharp_kernel  = sharpening_kernel(sigma)

zeros = np.zeros_like(B)


B_color = cv2.merge([B, zeros, zeros])
G_color = cv2.merge([zeros, G, zeros])
R_color = cv2.merge([zeros, zeros, R])


B_smooth = cv2.filter2D(B, -1, smooth_kernel)
G_smooth = cv2.filter2D(G, -1, smooth_kernel)
R_smooth = cv2.filter2D(R, -1, smooth_kernel)

img_smooth = cv2.merge([B_smooth, G_smooth, R_smooth])

B_smooth_color = cv2.merge([B_smooth, zeros, zeros])
G_smooth_color = cv2.merge([zeros, G_smooth, zeros])
R_smooth_color = cv2.merge([zeros, zeros, R_smooth])


B_sharp = cv2.filter2D(B, -1, sharp_kernel)
G_sharp = cv2.filter2D(G, -1, sharp_kernel)
R_sharp = cv2.filter2D(R, -1, sharp_kernel)

img_sharp = cv2.merge([B_sharp, G_sharp, R_sharp])

B_sharp_color = cv2.merge([B_sharp, zeros, zeros])
G_sharp_color = cv2.merge([zeros, G_sharp, zeros])
R_sharp_color = cv2.merge([zeros, zeros, R_sharp])


cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.imshow("Blue Channel", B_color)
cv2.waitKey(0)
cv2.imshow("Green Channel", G_color)
cv2.waitKey(0)
cv2.imshow("Red Channel", R_color)
cv2.waitKey(0)

cv2.imshow("Gausian Blue", B_smooth_color)
cv2.waitKey(0)
cv2.imshow("Gausian Green", G_smooth_color)
cv2.waitKey(0)
cv2.imshow("Gausian Red", R_smooth_color)
cv2.waitKey(0)
cv2.imshow("Merged Image", img_smooth)
cv2.waitKey(0)

cv2.imshow("Sharpened Blue", B_sharp_color)
cv2.waitKey(0)
cv2.imshow("Sharpened Green", G_sharp_color)
cv2.waitKey(0)
cv2.imshow("Sharpened Red", R_sharp_color)
cv2.waitKey(0)
cv2.imshow("Sharpened Image", img_sharp)

cv2.waitKey(0)
cv2.destroyAllWindows()
