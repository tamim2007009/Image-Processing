# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:21:26 2025

@author: USER
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cmath # Complex math library

# --- Core Functions (Implemented from scratch) ---

def dft_1d(signal, inverse=False):
    """
    Computes the 1D Discrete Fourier Transform of a signal.
    This is the raw implementation based on the DFT formula.
    Complexity: O(N^2)
    """
    N = len(signal)
    transform = np.zeros(N, dtype=np.complex128)
    
    for k in range(N):
        sum_val = 0
        for n in range(N):
            angle = -2 * cmath.pi * k * n / N
            if inverse:
                angle = -angle # Use positive exponent for inverse
            
            sum_val += signal[n] * cmath.exp(1j * angle)
        
        transform[k] = sum_val

    if inverse:
        return transform / N # Normalize for inverse
    
    return transform

def dft_2d(image, inverse=False):
    """
    Computes the 2D DFT by applying 1D DFT to each row, then each column.
    """
    M, N = image.shape
    # Apply 1D DFT to each row
    rows_transformed = np.zeros((M, N), dtype=np.complex128)
    for i in range(M):
        rows_transformed[i, :] = dft_1d(image[i, :], inverse)
        
    # Apply 1D DFT to each column of the row-transformed matrix
    final_transform = np.zeros((M, N), dtype=np.complex128)
    for j in range(N):
        final_transform[:, j] = dft_1d(rows_transformed[:, j], inverse)
        
    return final_transform
    
def create_notch_reject_filter(shape, points, radius):
    """
    Creates a Notch reject filter mask.
    It's 1 everywhere except for circles of radius 'r' around the specified points.
    """
    P, Q = shape
    H = np.ones((P, Q))
    
    # Create a grid of coordinates
    u, v = np.meshgrid(np.arange(Q), np.arange(P))
    
    # Center of the frequency domain
    center_u, center_v = Q // 2, P // 2
    
    for u_k, v_k in points:
        # Calculate distance from the first point and its symmetric counterpart
        D_k = np.sqrt((u - u_k)**2 + (v - v_k)**2)
        D_minus_k = np.sqrt((u - (Q - u_k))**2 + (v - (P - v_k))**2)
        
        # Set the filter to 0 inside the radius for both notches
        H[D_k <= radius] = 0
        H[D_minus_k <= radius] = 0
        
    return H

# --- Main Script ---

if __name__ == "__main__":
    # Load a base image (e.g., create a simple grayscale image)
    try:
        # You can replace this with your own image
        base_img = Image.open(r"two_noise.jpeg").convert('L') 
        base_img = np.array(base_img)
    except FileNotFoundError:
        print("lena.png not found. Creating a dummy 256x256 gray image.")
        base_img = np.full((256, 256), 128, dtype=np.uint8)

    M, N = base_img.shape

    # Create periodic noise
    x, y = np.meshgrid(np.arange(N), np.arange(M))
    noise = 50 * (np.sin(0.5 * x) + np.sin(0.5 * y))
    noisy_img = np.clip(base_img + noise, 0, 255).astype(np.uint8)

    # --- Start of Filtering Process (Following the Slides) ---

    # Step 1: Get padding parameters P and Q
    P, Q = 2 * M, 2 * N

    # Step 2: Form the padded image
    padded_img = np.zeros((P, Q))
    padded_img[0:M, 0:N] = noisy_img

    # Step 3: Multiply by (-1)^(x+y) to center the transform
    center_transform = np.zeros((P, Q))
    for i in range(P):
        for j in range(Q):
            center_transform[i, j] = padded_img[i, j] * ((-1)**(i+j))

    # Step 4: Compute the DFT
    print("Computing 2D DFT from scratch... This will take a while.")
    F_uv = dft_2d(center_transform)
    print("DFT computation complete.")

    # Show the power spectrum (for selecting coordinates)
    power_spectrum = np.log(1 + np.abs(F_uv))
    
    # Step 5: Generate the filter and apply it
    # These coordinates are identified by looking at the power spectrum.
    # The noise I created will produce spikes around (32,0) and (0,32) from the center.
    # Since the image is 512x512, the center is (256, 256).
    # So the points are (256+32, 256) and (256, 256+32) etc.
    # For this example, let's target the major spikes.
    # NOTE: The exact coordinates depend on the noise frequency. You find these
    # by inspecting the power_spectrum image.
    notch_center_1 = (256, 224) 
    notch_center_2 = (224, 256) 
    
    H_uv = create_notch_reject_filter((P, Q), points=[notch_center_1, notch_center_2], radius=15)
    
    # Apply the filter (element-wise multiplication)
    G_uv = F_uv * H_uv

    # Step 6: Compute the inverse DFT
    print("Computing Inverse 2D DFT from scratch... This will also take a while.")
    g_p = dft_2d(G_uv, inverse=True)
    print("Inverse DFT computation complete.")
    
    # Take the real part and multiply by (-1)^(x+y)
    for i in range(P):
        for j in range(Q):
            g_p[i, j] = g_p[i, j].real * ((-1)**(i+j))

    # Step 7: Extract the final M x N image
    final_img = g_p[0:M, 0:N]
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)

    # --- Display all results ---
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('1. Original Noisy Image')

    plt.subplot(2, 3, 2)
    plt.imshow(power_spectrum, cmap='gray')
    plt.title('2. Power Spectrum (Noise is visible)')

    plt.subplot(2, 3, 3)
    plt.imshow(H_uv, cmap='gray')
    plt.title('3. Notch Reject Filter H(u,v)')

    plt.subplot(2, 3, 4)
    plt.imshow(np.log(1 + np.abs(G_uv)), cmap='gray')
    plt.title('4. Spectrum after Filtering')

    plt.subplot(2, 3, 5)
    plt.imshow(final_img, cmap='gray')
    plt.title('5. Final Processed Image')
    
    plt.tight_layout()
    plt.show()