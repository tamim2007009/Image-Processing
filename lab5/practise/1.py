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
    img=cv2.imread('two_noise.jpeg',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('input',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Forward transform
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)

    magnitude_spectrum_ac = np.abs(ft_shift)
    magnitude_spectrum = 20 * np.log( magnitude_spectrum_ac + 1 ) # +1 to remove 0, 20 = scaling factor
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    phase = np.angle(ft_shift)
    phase_ = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    # phase add - dummy
    # final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*phase))

    # Process
    # magnitude_sp_ac * notch_filter

    h, w = img.shape

    indices = ( (272,256), (262,261) )
    radius = (5,5)
    
    notch = np.ones(img.shape)
    
    print(notch.shape)
    count = 0
    for u in range(h):
        for v in range(w):
            
            for i in range( len(indices) ):
                uk1 = indices[i][1]
                vk1 = indices[i][0]
                
                uk2 = h//2 - (uk1 - h//2)
                vk2 = w//2 - (vk1 - w//2)
                
                dist1 = math.sqrt( (u - uk1) ** 2 + (v - vk1)**2 )
                dist2 = math.sqrt( (u - uk2) ** 2 + (v - vk2)**2 )
                
                # uk1 -= h//2
                # vk1 -= w//2
                # dist1 = math.sqrt( (u - h/2 - uk1) ** 2 + (v -w/2 - vk1)**2 )
                # dist2 = math.sqrt( (u - h/2 + uk1) ** 2 + (v -w/2 + vk1)**2 )
                
                if( dist1 <= radius[i] or dist2 <= radius[i]):
                    count +=1 
                    notch[u,v] = 0
                    break


    print(count)
    # plt.imshow(notch)
    cv2.imshow("Notch", notch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return
    # plt.show()
    # print(notch)

    # Inverse transform
    
    mul = magnitude_spectrum_ac * notch
    #cv2.imshow("Multi",mul)
    
    final_result = np.multiply(mul, np.exp(1j*phase))
    center_shifted = np.fft.ifftshift(final_result)
    back_image = np.real( np.fft.ifft2(center_shifted) )
    back_image_scaled = cv2.normalize(back_image, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    ## plot
    cv2.imshow("input", img)
    cv2.imshow("Magnitude Spectrum",magnitude_spectrum)
    cv2.imshow("Phase", phase_)
    cv2.waitKey(0)

    cv2.imshow("Inverse transform",back_image_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

start()
