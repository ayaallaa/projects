# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.signal import convolve2d
from PIL import Image
#img = cv2.imread('girlWithScarf.png',0)
############################################## 1-Noise #######################################################
"""Uniform_Noise"""
def im_w_uniform_noise(low, high, im):
    u_noise=np.random.uniform(low, high, im.shape)
    noisy_im=im+u_noise;
    print("done")
    return noisy_im;  

#image_Noise = im_w_uniform_noise(0,100,im)  ### Qt
#fig = plt.figure(figsize=(10,10)) 
#plt.imshow(image_Noise,cmap='gray')

"""Gaussian_Noise"""
def im_gaussian_noise(mu, sigma, im):
    g_noise= np.random.normal(mu,sigma, im.shape)
    img_w_g_noise = im + g_noise
    return img_w_g_noise

#image_Noise = im_gaussian_noise(0, 100, im )  ### Qt

"""Salt&paper_Noise"""
def salt_pepper_noise(img,percent):
    img_noisy=np.zeros(img.shape)
    salt_pepper = np.random.random(img.shape) # Uniform distribution

    cleanPixels_ind=salt_pepper > percent
    pepper = (salt_pepper <= (0.5* percent)) # pepper < half percent
    salt = ((salt_pepper <= percent) & (salt_pepper > 0.5* percent)); 
    
    img_noisy[cleanPixels_ind]=img[cleanPixels_ind]
    img_noisy[pepper] = 0
    img_noisy[salt] = 1
    return img_noisy

#image_Noise = salt_pepper_noise(im, 0.1)  ### Qt 
########################################### 2-Filters #####################################################
"""Average_Filter"""
def averageFilter(image_Noise):  
  width = image_Noise.shape[1]
  height = image_Noise.shape[0]
  result = np.zeros((image_Noise.shape[0], image_Noise.shape[1]),dtype='uint8')
  for row in range(height):
     for col in range(width):  
         currentElement=0; left=0; right=0; top=0; bottom=0; topLeft=0; 
         topRight=0; bottomLeft=0; bottomRight=0;
         counter = 1           
         currentElement = image_Noise[row][col]

         if not col-1 < 0:
             left = image_Noise[row][col-1]
             counter +=1                        
         if not col+1 > width-1:
             right = image_Noise[row][col+1]
             counter +=1 
         if not row-1 < 0:
             top = image_Noise[row-1][col]
             counter +=1 
         if not row+1 > height-1:
             bottom = image_Noise[row+1][col]
             counter +=1 
         if not row-1 < 0 and not col-1 < 0:
             topLeft = image_Noise[row-1][col-1]
             counter +=1 
         if not row-1 < 0 and not col+1 > width-1:
             topRight = image_Noise[row-1][col+1]
             counter +=1 
         if not row+1 > height-1 and not col-1 < 0:
             bottomLeft = image_Noise[row+1][col-1]
             counter +=1 
         if not row+1 > height-1 and not col+1 > width-1:
             bottomRight = image_Noise[row+1][col+1]
             counter +=1
             
         total = int(currentElement)+int(left)+int(right)+int(top)+int(bottom)+int(topLeft)+int(topRight)+int(bottomLeft)+int(bottomRight)
         avg = total/counter
         result[row][col] = avg
  return result                
#averageFilter()  ### Qt        
#fig = plt.figure(figsize=(10,10)) 
#plt.imshow(result,cmap='gray')

"""Gussian_Filter"""
def gaussian_Filter(sigma, shape ):
    filter=np.zeros(shape)
    filter[1,1]=1
    return ndimage.gaussian_filter(filter, sigma) 

#smoothed_gFilter=convolve2d(image_Noise, gaussian_Filter(0.3,(3,3)),mode='same')  ### Qt
#smoothed_boxFilter=signal.convolve2d(image_Noise, \
#                                     [(1/9,1/9,1/9),(1/9,1/9,1/9),(1/9,1/9,1/9)], mode='same')  ### Qt

"""Median_Filter"""
#median = ndimage.median_filter(image_Noise,(3,3))  ### Qt
def median_filter(im, kernal):
    rows, cols= im.shape
    arr = []
    index = kernal // 2
    img_filtered = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            for z in range(kernal):
                if i + z - index < 0 or i + z - index > rows - 1:
                    for c in range(kernal):
                        arr.append(0)
                elif j + z - index < 0 or j + index > cols - 1:
                        arr.append(0)
                else:
                    for k in range(kernal):
                        arr.append(im[i + z - index][j + k - index])

            arr.sort()
            img_filtered[i][j] = arr[len(arr) // 2]
            arr = []
    return img_filtered   ### Qt     img_filtered(img_noisy,3)

############################################# 3-Edge_Detection ###############################################
"""roberts"""
#def load_image( infilename ) :
#    img = cv2.imread('apple.jpg', cv2.IMREAD_GRAYSCALE)
#    # note signed integer
#    return np.asarray( img, dtype="int32" )
#
#def save_image( data, outfilename ) :
#    img = Image.fromarray( np.asarray( np.clip(data,0,255), dtype="uint8"), "L" )
#    img.save( outfilename )
def roberts_cross( image ) :
#    image = load_image( infilename )
    roberts_cross_v = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

    roberts_cross_h = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )

    vertical = ndimage.convolve( image, roberts_cross_v )
    horizontal = ndimage.convolve( image, roberts_cross_h )

    output_image = np.sqrt( np.square(horizontal) + np.square(vertical))

    return output_image  ### Qt
#def main():
#    
#    Input_Image = cv2.imread("apple.jpg",0) # This is to read Gray Scale Image
#    
#    
#    image=roberts_cross( Input_Image, save_image )
#    cv2.imwrite('robert.jpg',  image)
#   
#    
#	#input("Please Enter to Continue...")
#   # input("Please Enter to Continue...")
#if __name__ == '__main__':
#	main()
#
"""Prewitt"""
def Prewitt (im):
       Gx = np.array([[1,0,-1],
                      [1,0,-1],
                      [1,0,-1]])
       Gy= np.array([[1,1,1],
                     [0,0,0],
                     [-1,-1,-1]])

       img_prewittx=convolve2d(im,Gx,mode='same')
       img_prewitty=convolve2d(im,Gy,mode='same')
       #Gradient_Magnitude
       out = np.sqrt(np.power(img_prewittx,2) + np.power(img_prewitty,2))
       return out 
##Prewitt ()  ### Qt      

##sobel 
#sample=cv2.imread("/Users/hp/Desktop/CV task1/chair.jpg",0)
       
"""Sobel"""
def sobel (im):
       kernal_y=np.zeros(shape=(3,3))
       kernal_y[0,0]=-1
       kernal_y[0,1]=-2
       kernal_y[0,2]=-1
       kernal_y[1,0]=0
       kernal_y[1,1]=0
       kernal_y[1,2]=0
       kernal_y[2,0]=1
       kernal_y[2,1]=2
       kernal_y[2,2]=1
       
       kernal_x=np.zeros(shape=(3,3))
       kernal_x[0,0]=-1
       kernal_x[0,1]=0
       kernal_x[0,2]=1
       kernal_x[1,0]=-1
       kernal_x[1,1]=0
       kernal_x[1,2]=1
       kernal_x[2,0]=-2
       kernal_x[2,1]=0
       kernal_x[2,2]=2
       
       gy=convolve2d(im,kernal_y)
       gx =convolve2d(im,kernal_x)
       G=np.hypot(gx, gy)
       theta = np.arctan2(gy, gx)
       g_sobel =norm(gx,gy)
       
       return g_sobel , G , theta
#cv2.imshow("gradient_y",gy)
#kernal=np.ndarray.flatten(kernal_y)#convert to 1D array so we can use convolve function
#sample=np.ndarray.flatten(sample)#convert to 1D array so we can use convolve function
#kernal=np.ndarray.flatten(kernal_x)
#cv2.imshow("gradient x",gx)
#cv2.imshow("sobel_edge",g_sobel)
def norm(img1,img2):
    im_copy=np.zeros(img1.shape)#images with initial zero values
    for i in  range(img1.shape[0]):
        for j in range(img1.shape[1]):
            q=(img1[i][j]**2 +img2[i][j]**2)**(1/2)
            if(q>90):#threshold
                im_copy[i][j]=255 #obtaining a binary image
            else:
                im_copy[i][j]=0         
    return im_copy

"""Canny"""
def Canny(img ,lowthreshold ,highthreshold ):
#    for i, img in enumerate(self.imgs):    
     img_smoothed = convolve2d(img, gaussian_Filter(0.3,(3,3)),mode='same')   
     FilterImg , G , theta =sobel( img_smoothed )
     image_edge=non_max_suppression(FilterImg, G , theta)
     new_img=threshold( image_edge ,lowthreshold ,highthreshold )
     final_img=hysteresis(new_img)
     return final_img
#(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15)
       
def non_max_suppression(FilterImg, G , theta):
        R, C = FilterImg.shape
        img = np.zeros((R,C), dtype=np.int32)
        G = G / G.max() * 255
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        
        for i in range(1,R-1):
            for j in range(1,C-1):
                #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = G[i, j+1]
                        r = G[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = G[i+1, j-1]
                        r = G[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = G[i+1, j]
                        r = G[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = G[i-1, j-1]
                        r = G[i+1, j+1]

                    if (G[i,j] >= q) and (G[i,j] >= r):
                        img[i,j] = G[i,j]
                    else:
                        img[i,j] = 0
        img = img / img.max() * 255   
        print(img.max())
        return img
                
strong_pexel=255
weak_pexcel=140          
def threshold( img ,lowthreshold ,highthreshold ):
        M, N = img.shape
        new_img = np.zeros((M,N), dtype=np.int32)
        for i in range(1, M-1):
            for j in range(1, N-1):
                if(img[i,j] >= highthreshold):
                    new_img[i,j] = strong_pexel
                elif(img[i,j] <= lowthreshold):
                    new_img[i,j] = 0
                elif(img[i,j] > lowthreshold) and (img[i,j] < highthreshold) :
                    new_img[i,j] = weak_pexcel

        return new_img
def hysteresis( img):
        M, N = img.shape

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak_pexcel):
                        if ((img[i+1, j-1] == strong_pexel) or (img[i+1, j] == strong_pexel) or (img[i+1, j+1] == strong_pexel)
                            or (img[i, j-1] == strong_pexel) or (img[i, j+1] == strong_pexel)
                            or (img[i-1, j-1] == strong_pexel) or (img[i-1, j] == strong_pexel) or (img[i-1, j+1] == strong_pexel)):
                            img[i, j] = strong_pexel
                        else:
                            img[i, j] = 0  
        return img