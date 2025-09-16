import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

# -------- Step 1: Load Lena image in grayscale --------
img = Image.open("two_noise.jpeg").convert("L")   # grayscale
img = np.array(img, dtype=float)
rows, cols = img.shape

# -------- Step 2: 1D DFT --------
def dft1d(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        s = 0.0
        for n in range(N):
            angle = -2j * math.pi * k * n / N
            s += signal[n] * np.exp(angle)
        X[k] = s
    return X

# -------- Step 3: 1D IDFT --------
def idft1d(signal):
    N = len(signal)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        s = 0.0
        for k in range(N):
            angle = 2j * math.pi * k * n / N
            s += signal[k] * np.exp(angle)
        x[n] = s / N
    return x

# -------- Step 4: 2D DFT using separable 1D --------
def dft2d(image):
    M, N = image.shape
    # DFT on rows
    temp = np.zeros((M, N), dtype=complex)
    for i in range(M):
        temp[i, :] = dft1d(image[i, :])
    # DFT on columns
    F = np.zeros((M, N), dtype=complex)
    for j in range(N):
        F[:, j] = dft1d(temp[:, j])
    return F

# -------- Step 5: 2D IDFT using separable 1D --------
def idft2d(F):
    M, N = F.shape
    # IDFT on columns
    temp = np.zeros((M, N), dtype=complex)
    for j in range(N):
        temp[:, j] = idft1d(F[:, j])
    # IDFT on rows
    f = np.zeros((M, N), dtype=complex)
    for i in range(M):
        f[i, :] = idft1d(temp[i, :])
    return f.real

# -------- Step 6: Power Spectrum --------
def power_spectrum(F):
    return np.log(1 + np.abs(F))

# -------- Step 7: Notch Reject Filter --------
def notch_reject_filter(shape, centers, radius):
    M, N = shape
    H = np.ones((M, N))
    for (u0, v0) in centers:
        for u in range(M):
            for v in range(N):
                D = math.sqrt((u - u0)**2 + (v - v0)**2)
                if D <= radius:
                    H[u, v] = 0
    return H

# -------- Run pipeline --------
print("Computing 2D DFT (faster separable version)...")
F = dft2d(img)
P = power_spectrum(F)

# Example: notch positions (manually chosen from spectrum)
notches = [(rows//2 + 30, cols//2 + 30), (rows//2 - 30, cols//2 - 30)]
radius = 10
H = notch_reject_filter(img.shape, notches, radius)

# Apply filter
G = F * H

print("Computing inverse 2D DFT...")
img_filtered = idft2d(G)

# -------- Show results --------
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(132), plt.imshow(P, cmap='gray'), plt.title("Power Spectrum")
plt.subplot(133), plt.imshow(img_filtered, cmap='gray'), plt.title("Filtered")
plt.show()
