import numpy as np
import cv2

# Laplacian of Gaussian Kernel
def log_kernel(sigma):
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1

    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    xr, yc = np.meshgrid(x, y)

    norm = (xr**2 + yc**2 - 2 * sigma**2) / (sigma**4)
    kernel = norm * np.exp(-(xr**2 + yc**2) / (2 * sigma**2))
    kernel = kernel - np.mean(kernel)
    return kernel


# Zero Crossing
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


# Local Variance in window
def local_variance(img, ksize=7):
    mean = cv2.blur(img, (ksize, ksize))
    sqr_mean = cv2.blur(img**2, (ksize, ksize))
    variance = sqr_mean - mean**2
    return variance


# Robust LoG Edge Detector
def robust_log_edge(img, sigma=1, var_thresh=60, win_size=7):
    # Step 1: LoG filtering
    log = log_kernel(sigma)
    img_log = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=log)

    # Step 2: Zero Crossing
    zc = zero_crossing(img_log)

    # Step 3: Local Variance
    var_map = local_variance(img.astype(np.float32), ksize=win_size)

    # Step 4: Keep edges only if variance > threshold
    robust_edges = np.zeros_like(zc)
    robust_edges[(zc == 255) & (var_map > var_thresh)] = 255

    return img_log, zc, robust_edges


# ---------------- Main ----------------
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
sigma = float(input("Enter sigma value: "))

cv2.imshow("Original", img)
cv2.waitKey(0)

img_log, zc, robust_edges = robust_log_edge(img, sigma=sigma, var_thresh=60, win_size=7)

# Normalize LoG for display
log_norm = np.round(cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
cv2.imshow("LoG Response", log_norm)
cv2.waitKey(0)

cv2.imshow("Zero Crossing", zc)
cv2.waitKey(0)

cv2.imshow("Robust Laplacian Edges", robust_edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
