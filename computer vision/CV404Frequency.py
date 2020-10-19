# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 

def HPF (im1):
    global img_HPF 
    f1 = np.fft.fft2(im1)
    fshift1 = np.fft.fftshift(f1)
    r1, c1 = im1.shape
    mask1=[int(r1/2), int(c1/2)]
    fshift1[mask1[0]-10:mask1[0]+10, mask1[1]-10:mask1[1]+10] = 0
    sh_inverse1 = np.fft.ifftshift(fshift1)
    img_HPF= np.abs(np.fft.ifft2(sh_inverse1))
    return img_HPF

def LPF (im2):
    global img_LPF
    f2 = np.fft.fft2(im2)
    r2, c2 = im2.shape
    remain= 0.1
    f2[int(r2*remain):int(r2*(1-remain))] = 0      # Rows
    f2[:,int(c2*remain):int(c2*(1-remain))] = 0    # Columns
    img_LPF= np.abs(np.fft.ifft2(f2))
    return img_LPF

def Hybrid():  
    result= img_HPF+img_LPF
    return(result)