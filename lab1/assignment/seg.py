import numpy as np
import cv2

# Gaussian Kernel
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


# Laplacian of Gaussian (LoG) Kernel
def log_kernel(sigma):
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1

    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    xr, yc = np.meshgrid(x, y)

    norm = (xr**2 + yc**2 - 2 * sigma**2) / (sigma**4)
    kernel = norm * np.exp(-(xr**2 + yc**2) / (2 * sigma**2))

    kernel = kernel - np.mean(kernel)   # normalize to zero mean
    return kernel


# Zero Crossing Detection
def zero_crossing(img):
    zc_img = np.zeros(img.shape, dtype=np.uint8)
    rows, cols = img.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            patch = img[i-1:i+2, j-1:j+2]
            p = img[i, j]
            if (p > 0 and np.min(patch) < 0) or (p < 0 and np.max(patch) > 0):
                zc_img[i, j] = 255
    return zc_img


# Thresholding after Zero Crossing
def threshold_edges(img, thresh=60):
    edges = np.zeros_like(img)
    edges[img > thresh] = 255
    return edges


# ---------------------- Main Program ----------------------
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

sigma = float(input("Enter sigma value: "))

# Original Image
cv2.imshow("Original", img)
cv2.waitKey(0)

# Convolve with LoG kernel
log = log_kernel(sigma)
img_log = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=log)

# Normalize for display
log_norm = np.round(cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
cv2.imshow("LoG Response", log_norm)
cv2.waitKey(0)

# Zero Crossing
zc = zero_crossing(img_log)
cv2.imshow("Zero Crossing", zc)
cv2.waitKey(0)

# Thresholding on LoG response
thresh_img = threshold_edges(np.abs(img_log), thresh=60)
cv2.imshow("Thresholded Edges", thresh_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
