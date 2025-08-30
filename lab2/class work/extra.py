import numpy as np
import matplotlib.pyplot as plt
import cv2

def gaussian_derivative_kernels(sigma, normalize=True):
    size = int(7 * sigma)
    if size % 2 == 0:
        size += 1

    x = np.arange(-size//2, size//2 + 1)
    y = np.arange(-size//2, size//2 + 1)
    X, Y = np.meshgrid(x, y)

    const = 1.0 / (2.0 * np.pi * (sigma ** 2))
    gaussian = const * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))

    # Derivatives
    Gx = -(X / (sigma**2)) * gaussian
    Gy = -(Y / (sigma**2)) * gaussian

    if normalize:
        def normalize_kernel(K):
            K_norm = cv2.normalize(K, None, alpha=-25, beta=25, norm_type=cv2.NORM_MINMAX)
            return np.round(K_norm).astype(int)   # <-- return added

        Gx = normalize_kernel(Gx)
        Gy = normalize_kernel(Gy)

    return Gx, Gy


# Example with sigma = 1
sigma = 1
Gx, Gy = gaussian_derivative_kernels(sigma)

print("Normalized X-derivative kernel:\n", Gx)
print("Normalized Y-derivative kernel:\n", Gy)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("X-derivative kernel")
plt.imshow(Gx, cmap="gray", interpolation="nearest")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Y-derivative kernel")
plt.imshow(Gy, cmap="gray", interpolation="nearest")
plt.colorbar()

plt.show()
