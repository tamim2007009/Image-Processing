 # -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 07:26:48 2025
@author: USER
"""

import numpy as np
import cv2
import math

def start():
    # ---------- Step 1: Read Image ----------
    img = cv2.imread('two_noise.jpeg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found!")
        return
    
    cv2.imshow('Input Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------- Step 2: Fourier Transform ----------
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)

    # ---------- Step 3: Magnitude and Phase ----------
    magnitude_spectrum_ac = np.abs(ft_shift)
    magnitude_spectrum = 20 * np.log(magnitude_spectrum_ac + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255,
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    phase = np.angle(ft_shift)
    phase_ = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ---------- Step 4: Define Image Shape ----------
    h, w = img.shape

    # ---------- Step 5: Define ONE Noise Peak ----------
    vk1, uk1 = 272, 256   # (row, col)
    r = 5                 # radius

    # Compute symmetric point (mirror across center)
    vk2 = w//2 - (vk1 - w//2)
    uk2 = h//2 - (uk1 - h//2)

    print(f"Primary notch: ({vk1},{uk1})")
    print(f"Symmetric notch: ({vk2},{uk2})")

    # ---------- Step 6: Create Notch Filter ----------
    notch = np.ones((h, w), dtype=np.float32)

    for u in range(h):
        for v in range(w):
            dist1 = math.sqrt((u - uk1)**2 + (v - vk1)**2)
            dist2 = math.sqrt((u - uk2)**2 + (v - vk2)**2)

            if dist1 <= r or dist2 <= r:
                notch[u, v] = 0

    cv2.imshow("Notch Filter", notch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------- Step 7: Apply Filter ----------
    filtered_magnitude = magnitude_spectrum_ac * notch
    final_result = np.multiply(filtered_magnitude, np.exp(1j * phase))

    center_shifted = np.fft.ifftshift(final_result)
    back_image = np.real(np.fft.ifft2(center_shifted))
    back_image_scaled = cv2.normalize(back_image, None, 0, 255,
                                      cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ---------- Step 8: Display Results ----------
    cv2.imshow("Input Image", img)
    cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
    cv2.imshow("Phase", phase_)
    cv2.imshow("Filtered Image (After Notch)", back_image_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

start()
