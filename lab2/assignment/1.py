import numpy as np
import cv2

def gaussian_derivative_kernels(sigma):
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1
    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    X, Y = np.meshgrid(x, y)
    const = 1.0 / (2.0 * np.pi * (sigma ** 2))
    gaussian = const * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    Gx = -(X / (sigma**2)) * gaussian
    Gy = -(Y / (sigma**2)) * gaussian
    return Gx, Gy

def hysteresis_thresholding(grad, low, high):
    """Apply hysteresis thresholding to a gradient image."""
    strong = 255
    weak = 128
    output = np.zeros_like(grad, dtype=np.uint8)

    # Step 1: classify pixels as strong, weak, or zero
    strong_pixels = grad >= high
    weak_pixels = (grad >= low) & (grad < high)

    output[strong_pixels] = strong
    output[weak_pixels] = weak

    # Step 2: Track edges by connecting weak pixels to strong ones
    h, w = grad.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if output[i, j] == weak:
                # Check 8-connected neighbors
                if np.any(output[i-1:i+2, j-1:j+2] == strong):
                    output[i, j] = strong
                else:
                    output[i, j] = 0
    return output

# --- Main Script ---
sigma = float(input("Enter the value of sigma: "))
gx, gy = gaussian_derivative_kernels(sigma)

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

xcon = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=gx)
ycon = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=gy)

grad_mag = np.sqrt(xcon**2 + ycon**2)
grad_mag = np.round(cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow("Original Gradient Magnitude", grad_mag)
cv2.waitKey(0)

low, high = map(int, input("Enter low and high thresholds (e.g., 50 150): ").split())
edges = hysteresis_thresholding(grad_mag, low, high)

cv2.imshow("Edges after Hysteresis Thresholding", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
