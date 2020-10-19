# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import ndimage, signal
import cv2 as cv

def Harris_corner_detector (img):
    global R, w, h, RGB_img, height, width,imggray, Ixx, Iyy, Ixy
    RGB_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    imggray = np.copy(img)
    height, width = RGB_img.shape[:2]
    w, h = img.shape
    ## Gussian_Filter  
    filter = np.zeros((3,3))
    filter[1,1]=1
    gaussian = ndimage.gaussian_filter(filter, 0.7)
    img_smoothed = signal.convolve2d(img,gaussian,mode='same')

    ## Sobel operator kernels.
    Gx = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])

    Gy = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])   

    Ix = signal.convolve2d( img_smoothed , Gx ,'same')
    Iy = signal.convolve2d( img_smoothed , Gy ,'same')
    
    ## Construct the Hessian (Hesh'n) matrix M
    Ixx =  np.multiply(Ix,Ix) 
    Iyy =  np.multiply(Iy,Iy)
    Ixy =  np.multiply(Ix,Iy)
    
    ## Construct the Hessian (Hesh'n) matrix M over a window
    window = np.array([(1/9,1/9,1/9),
                       (1/9,1/9,1/9),
                       (1/9,1/9,1/9)])
    
    Ixx_hat = signal.convolve2d( Ixx , window ,'same') 
    Iyy_hat = signal.convolve2d( Iyy , window ,'same') 
    Ixy_hat = signal.convolve2d( Ixy , window ,'same')
    
    ## Evaluating R
    K = 0.06
    detM = np.multiply(Ixx_hat,Iyy_hat) - np.multiply(Ixy_hat,Ixy_hat) 
    trM = Ixx_hat + Iyy_hat
    R = detM - K * (trM**2)  
    return (R, Ixx, Iyy, Ixy) 

#### Corners detection different techniques #### 
""" 1. Thresholding by constant absolute value """
def Thresholding (r,th):
    global R, w, h
    radius = 1
    color = (255,0,0)  # Green
    thickness = 1
    # Look for Corner strengths above the threshold
    for row in range(w):
           for col in range(h):
              if R[row][col] > th:
                  cv2.circle(RGB_img, (col, row), radius, color, thickness)
    return (RGB_img)

""" 2. Local Thresholding """
def Local_Thresholding (I_xx, I_yy, I_xy):
     global RGB_img, height, width, imggray,Ixx, Iyy, Ixy
     height, width = RGB_img.shape[:2]
     window_size = 5
     offset = int(window_size/2)
     r = np.zeros(imggray.shape)
     
     for y in range(offset, height-offset):
        for x in range(offset, width-offset):
          window_Ixx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
          window_Iyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
          window_Ixy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]

          Mxx = window_Ixx.sum()
          Myy = window_Iyy.sum()
          Mxy = window_Ixy.sum()
          
          det = Mxx*Myy - Mxy*Mxy
          trace = Mxx + Myy
          r[y,x] = det - 0.04 * (trace ** 2)        

     for y in range(offset, height-offset):
         for x in range(offset, width-offset):
             if r[y, x] > 0.4:
                cv.circle(RGB_img, (x, y), 1, (255, 0, 0), 1)
     return (RGB_img)
          
""" 3. Non-maximum supression """
def Non_maximum_supression (r):
    global R, w, h
    for row in range(w):
           for col in range(h):              
                 skip = True
                 for nrow in range (1):
                    for ncol in range (1):
                       if row + nrow -1 < w and col + ncol -1 < h:
                          maxx = R[nrow][ncol]
                          if R [row+nrow -1][col + ncol -1 ] > maxx:
                             skip = False
                             break
                 if not skip:
                  cv2.circle(RGB_img, (col, row), 1, (255, 0,0), 1)
    return (RGB_img)