import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussianF(x, y, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def gaussKernel(size, sigma):
    k_half = size // 2
    coords = np.arange(-k_half, k_half + 1)
    x, y = np.meshgrid(coords, coords)
    kernel = gaussianF(x, y, sigma)
    kernel /= np.sum(kernel)
    return kernel

def gaussXkernel(size, sigma):
    k_half = size // 2
    coords = np.arange(-k_half, k_half + 1)
    x, y = np.meshgrid(coords, coords)
    gv = gaussianF(x, y, sigma)
    gx = -(x / sigma**2) * gv
    # FIX 1: Normalize properly - avoid division by zero
    sum_abs = np.sum(np.abs(gx))
    if sum_abs > 1e-10:
        gx /= sum_abs
    return gx

def gaussYkernel(size, sigma):
    k_half = size // 2
    coords = np.arange(-k_half, k_half + 1)
    x, y = np.meshgrid(coords, coords)
    gv = gaussianF(x, y, sigma)
    gy = -(y / sigma**2) * gv
    # FIX 1: Normalize properly - avoid division by zero
    sum_abs = np.sum(np.abs(gy))
    if sum_abs > 1e-10:
        gy /= sum_abs
    return gy

def logFunction(x, y, sigma):
    x2y2 = x**2 + y**2
    return -(1 / (np.pi * sigma**4)) * (1 - x2y2 / (2 * sigma**2)) * np.exp(-x2y2 / (2 * sigma**2))

def log_kernel(size, sigma):
    k_half = size // 2
    coords = np.arange(-k_half, k_half + 1)
    x, y = np.meshgrid(coords, coords)
    kernel = logFunction(x, y, sigma)
    # FIX 2: LoG should sum to zero (remove DC component)
    kernel = kernel - np.mean(kernel)
    return kernel

# Load image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if img is None:
    print("Error: Could not load image. Make sure 'lena.jpg' exists.")
    exit()

h, w = img.shape
print(f"Image loaded: {h}x{w}")

# FIX 3: Use reasonable default values
s = 5  # kernel size (use odd numbers: 3, 5, 7, 9)
sigma = 1.0  # sigma value

print(f"Using kernel size: {s}, sigma: {sigma}")

# Manual Gaussian convolution
pad = s // 2
img_bordered = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)  # FIX 4: Use REPLICATE
outGaussian = np.zeros((h, w), dtype=np.float64)
kerG = gaussKernel(s, sigma)

print("Performing manual Gaussian convolution...")
for i in range(h):
    for j in range(w):
        region = img_bordered[i:i+s, j:j+s]
        outGaussian[i, j] = np.sum(region * kerG)

outGaussian = cv2.normalize(outGaussian, None, 0, 255, cv2.NORM_MINMAX)
outGaussian = np.round(outGaussian).astype(np.uint8)

# Derivative filters and LoG
print("Computing derivative filters...")
kerGx = gaussXkernel(s, sigma)
kerGy = gaussYkernel(s, sigma)
kerLoG = log_kernel(s, sigma)

# Print kernels for verification
print("\nGaussian X Derivative Kernel:")
print(kerGx)
print("\nGaussian Y Derivative Kernel:")
print(kerGy)

# FIX 5: Use CV_64F for derivative filters to preserve negative values
outGx = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kerGx)
outGy = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kerGy)
outLoG = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kerLoG)

# FIX 6: Normalize derivative outputs for display
# For derivatives, we need to handle negative values
outGx_display = cv2.normalize(outGx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
outGy_display = cv2.normalize(outGy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
outLoG_display = cv2.normalize(outLoG, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Calculate gradient magnitude
gradient_magnitude = np.sqrt(outGx**2 + outGy**2)
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display results using matplotlib (better for seeing multiple images)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(outGaussian, cmap='gray')
axes[0, 1].set_title('Gaussian Blur')
axes[0, 1].axis('off')

axes[0, 2].imshow(gradient_magnitude, cmap='gray')
axes[0, 2].set_title('Gradient Magnitude')
axes[0, 2].axis('off')

axes[1, 0].imshow(outGx_display, cmap='gray')
axes[1, 0].set_title('Gaussian X Derivative')
axes[1, 0].axis('off')

axes[1, 1].imshow(outGy_display, cmap='gray')
axes[1, 1].set_title('Gaussian Y Derivative')
axes[1, 1].axis('off')

axes[1, 2].imshow(outLoG_display, cmap='gray')
axes[1, 2].set_title('Laplacian of Gaussian')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("\nResults saved to 'results.png'")
plt.show()

# Also display with OpenCV if you prefer
cv2.imshow('Original', img)
cv2.imshow('Gaussian Blur', outGaussian)
cv2.imshow('Gradient Magnitude', gradient_magnitude)
cv2.imshow('X Derivative', outGx_display)
cv2.imshow('Y Derivative', outGy_display)
cv2.imshow('Laplacian of Gaussian', outLoG_display)

print("\nPress any key to close windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()