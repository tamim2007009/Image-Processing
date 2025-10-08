# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 08:20:28 2025

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 07:26:48 2025
@author: USER
"""

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def start():
    # ---------- Step 1: Read Image ----------
    img = cv2.imread('two_noise.jpeg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Input Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------- Step 2: Forward Fourier Transform ----------
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)

    # ---------- Step 3: Magnitude & Phase ----------
    magnitude_spectrum_ac = np.abs(ft_shift)
    magnitude_spectrum = 20 * np.log(magnitude_spectrum_ac + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255,
                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    phase = np.angle(ft_shift)
    phase_ = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ---------- Step 4: Image Size ----------
    h, w = img.shape

    # ---------- Step 5: Define Noise Peaks (Indices) ----------
    # You can add as many as needed
    indices = [ (272,256), (262,261) ]
    radius = [5, 5]

    # ---------- Step 6: Initialize the Filter ----------
    notch = np.ones((h, w), dtype=np.float32)

    # ---------- Step 7: Process One Index at a Time ----------
    for i in range(len(indices)):
        uk1, vk1 = indices[i][1], indices[i][0]   # (col, row)
        r = radius[i]

        # Find symmetric point (mirror around center)
        uk2 = h//2 - (uk1 - h//2)
        vk2 = w//2 - (vk1 - w//2)

        print(f"\nProcessing notch {i+1}:")
        print(f"Primary point: ({vk1},{uk1}), Symmetric point: ({vk2},{uk2}), Radius: {r}")

        # Create temporary mask for this notch
        temp_mask = np.ones((h, w), dtype=np.float32)

        for u in range(h):
            for v in range(w):
                dist1 = math.sqrt((u - uk1) ** 2 + (v - vk1) ** 2)
                dist2 = math.sqrt((u - uk2) ** 2 + (v - vk2) ** 2)

                if dist1 <= r or dist2 <= r:
                    temp_mask[u, v] = 0

        # Multiply the overall mask by this notch
        notch = notch * temp_mask

        # Optionally visualize each individual notch
        cv2.imshow(f"Notch mask {i+1}", temp_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ---------- Step 8: Show Final Combined Notch ----------
    cv2.imshow("Combined Notch Filter", notch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------- Step 9: Apply Notch Filter ----------
    filtered_magnitude = magnitude_spectrum_ac * notch

    final_result = np.multiply(filtered_magnitude, np.exp(1j * phase))
    center_shifted = np.fft.ifftshift(final_result)
    back_image = np.real(np.fft.ifft2(center_shifted))
    back_image_scaled = cv2.normalize(back_image, None, 0, 255,
                                      cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ---------- Step 10: Display Results ----------
    cv2.imshow("Input Image", img)
    cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
    cv2.imshow("Phase", phase_)
    cv2.imshow("Inverse Transform (Filtered Image)", back_image_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

start()
