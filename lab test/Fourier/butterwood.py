# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 07:26:48 2025
@author: USER

Modified by ChatGPT (GPT-5)
Butterworth Notch Reject Filter Example
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

    h, w = img.shape
    crow, ccol = h // 2, w // 2

    # ---------- Step 4: Define Noise Peaks ----------
    vk1, uk1 = 272, 256  # given noise location
    D0 = 10              # cutoff frequency
    n = 2                # order of Butterworth filter

    # Mirror point
    vk2 = w // 2 - (vk1 - w // 2)
    uk2 = h // 2 - (uk1 - h // 2)

    print(f"Primary notch: ({vk1},{uk1})")
    print(f"Symmetric notch: ({vk2},{uk2})")

    # ---------- Step 5: Create Butterworth Notch Reject Filter ----------
    H = np.ones((h, w), dtype=np.float32)
    for u in range(h):
        for v in range(w):
            D1 = math.sqrt((u - uk1)**2 + (v - vk1)**2)
            D2 = math.sqrt((u - uk2)**2 + (v - vk2)**2)
            H[u, v] = 1 / (1 + ((D0**2 / (D1 * D2 + 1e-5))**n))  # avoid division by zero

    # Normalize and display filter
    H_disp = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Butterworth Notch Reject Filter", H_disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------- Step 6: Apply Filter ----------
    filtered_ft = ft_shift * H
    filtered_img = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_ft)))
    filtered_img_scaled = cv2.normalize(filtered_img, None, 0, 255,
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ---------- Step 7: Display Results ----------
    cv2.imshow("Input Image", img)
    cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
    cv2.imshow("Butterworth Notch Filter", H_disp)
    cv2.imshow("Filtered Image (Butterworth Notch Reject)", filtered_img_scaled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

start()
