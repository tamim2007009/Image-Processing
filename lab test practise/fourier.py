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

vk1 = int(input("Enter noise point column (vk1): "))
uk1 = int(input("Enter noise point row (uk1): "))

r=int(input("Enter radius:"))


vk2=w//2-(vk1-w//2)
uk2=h//2-(uk1-h//2)
 
notch=np.ones((h,w),dtype=np.float32)

for i in range (h):
    for j in range(w):
        
        dist1=math.sqrt((uk1-i)**2+(vk1-j)**2)
        dist2=math.sqrt((uk2-i)**2+(vk2-j)**2)
        
        if dist1<=r or dist2<=r:
            notch[i,j]=0


cv2.imshow("Notch filter",notch)
cv2.waitKey(0)


filter_mag=mag_ac*notch
final_image=np.multiply(filter_mag,np.exp(1j*phase))

center_shift=np.fft.ifftshift(final_image)
filter_image=np.real(np.fft.ifft2(center_shift))

back_img=np.round(cv2.normalize(filter_image,None,0,255,cv2.NORM_MINMAX)).astype(np.uint8)

cv2.imshow(" filtered image",back_img)
cv2.waitKey(0)


cv2.destroyAllWindows()
