# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 17:29:46 2025
@author: USER
"""

import numpy as np
import cv2
import math

img = cv2.imread('pnois2.jpg', 0)
cv2.imshow("Original Image", img)
cv2.waitKey(0)



ft=np.fft.fft2(img)
ft_shift=np.fft.fftshift(ft)


mag_ac=np.abs(ft_shift)
phase=np.angle(ft_shift)


mag=20*np.log(mag_ac+1)
mag_=np.round(cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)).astype(np.uint8)
phase_=np.round(cv2.normalize(phase,None,0,255,cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow("Magnetude",mag_)
cv2.waitKey(0)


cv2.imshow("Phase Image", phase_)
cv2.waitKey(0)


h,w=img.shape


uk1 = int(input("Enter noise point row (uk1): "))

vk1 = int(input("Enter noise point column (vk1): "))


vk2=w//2-(vk1-w//2)
uk2=h//2-(uk1-h//2)
 
d0=int(input("Enter cutoff frequency"))
n=int(input("Enter order"))

H=np.ones((h,w),dtype=np.float32)


for u in range (h):
    for v in range (w):
        
        d1=np.sqrt((u-uk1)**2+(v-vk1)**2)
        d2=np.sqrt((u-uk2)**2+(v-vk2)**2)
        
        H[u,v]=1/(1+(d0**2/(d1*d2+1e-7))**n)



cv2.imshow("Filte", H)
cv2.waitKey(0)







filter_mag=mag_ac*H
final_image=np.multiply(filter_mag,np.exp(1j*phase))

center_shift=np.fft.ifftshift(final_image)
filter_image=np.real(np.fft.ifft2(center_shift))

back_img=np.round(cv2.normalize(filter_image,None,0,255,cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow(" filtered image",back_img)
cv2.waitKey(0)


cv2.destroyAllWindows()
