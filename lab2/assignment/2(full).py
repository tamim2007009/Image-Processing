import numpy as np
import cv2

def gaussian_derivative_kernels(sigma):
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1 
    half = size // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)

    const = 1.0 / (2.0 * np.pi * (sigma ** 2))
    gaussian = const * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))

    Gx = -(X / (sigma**2)) * gaussian
    Gy = -(Y / (sigma**2)) * gaussian
    return Gx, Gy

def non_maximum_suppression_no_tan(grad_mag, gx, gy):
    """
    Non-maximum suppression without atan.
    We assign direction by comparing gx and gy.
    """
    h, w = grad_mag.shape
    output = np.zeros_like(grad_mag, dtype=np.float32)

    for i in range(1, h-1):
        for j in range(1, w-1):
            # Get derivatives
            gx_val = gx[i, j]
            gy_val = gy[i, j]
            mag = grad_mag[i, j]

            # Skip low magnitude
            if mag == 0:
                continue

            # Approximate direction
            if abs(gx_val) > abs(gy_val):
                if abs(gy_val) < 0.414 * abs(gx_val):   # slope < tan(22.5°)
                    # 0° (horizontal)
                    q = grad_mag[i, j+1]
                    r = grad_mag[i, j-1]
                else:
                    # 45° or 135°
                    if gx_val * gy_val > 0:   # positive slope
                        q = grad_mag[i-1, j+1]
                        r = grad_mag[i+1, j-1]
                    else:                     # negative slope
                        q = grad_mag[i+1, j+1]
                        r = grad_mag[i-1, j-1]
            else:
                if abs(gx_val) < 0.414 * abs(gy_val):  # slope > tan(67.5°)
                    # 90° (vertical)
                    q = grad_mag[i+1, j]
                    r = grad_mag[i-1, j]
                else:
                    # 45° or 135°
                    if gx_val * gy_val > 0:   # positive slope
                        q = grad_mag[i-1, j+1]
                        r = grad_mag[i+1, j-1]
                    else:                     # negative slope
                        q = grad_mag[i+1, j+1]
                        r = grad_mag[i-1, j-1]

            # Keep only local maxima
            if mag >= q and mag >= r:
                output[i, j] = mag
            else:
                output[i, j] = 0

    return output.astype(np.uint8)

def hysteresis_thresholding(grad, low, high):
    strong, weak = 255, 128
    output = np.zeros_like(grad, dtype=np.uint8)
    h, w = grad.shape

    # First pass
    for i in range(h):
        for j in range(w):
            if grad[i, j] >= high:
                output[i, j] = strong
            elif grad[i, j] >= low:
                output[i, j] = weak
            else:
                output[i, j] = 0

    # Track weak edges connected to strong
    for i in range(1, h-1):
        for j in range(1, w-1):
            if output[i, j] == weak:
                if np.any(output[i-1:i+2, j-1:j+2] == strong):
                    output[i, j] = strong
                else:
                    output[i, j] = 0

    return output

if __name__ == "__main__":
    sigma = float(input("Enter the value of sigma (e.g., 1.0): "))
    low, high = map(int, input("Enter low and high thresholds (e.g., 50 150): ").split())
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    # Gradient
    gx_kernel, gy_kernel = gaussian_derivative_kernels(sigma)
    Ix = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=gx_kernel)
    Iy = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=gy_kernel)

    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)

    grad_mag_display = np.round(cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
    cv2.imshow("1. Gradient Magnitude", grad_mag_display)
    cv2.waitKey(1)

    suppressed_img = non_maximum_suppression_no_tan(grad_mag_display, Ix, Iy)
    cv2.imshow("2. Non-Maximum Suppression (No Tan)", suppressed_img)
    cv2.waitKey(1)

    final_edges = hysteresis_thresholding(suppressed_img, low, high)
    cv2.imshow("3. Final Edge Map (Canny)", final_edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
