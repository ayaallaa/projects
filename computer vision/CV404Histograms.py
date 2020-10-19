import numpy as np	#This is to deal with numbers and arrays
import cv2 as cv	#This is to deal with images
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import CV404Filters as filters
from PIL import Image

def Max(Current_Value, New_Value):
	if New_Value > Current_Value:
		return New_Value
	else:
		return Current_Value
		
def Min(Current_Value, New_Value):
	if New_Value < Current_Value:
		return New_Value
	else:
		return Current_Value

def Histogram_Computation(Image):
	Image_Height = Image.shape[0]
	Image_Width = Image.shape[1]
	Histogram = np.zeros([256], np.int32)
	
	Max_Intensity = 0
	Min_Intensity = 255
	
	for x in range(0, Image_Height):
		for y in range(0, Image_Width):
			Histogram[Image[x,y]] +=1
			Max_Intensity = Max(Max_Intensity, Image[x,y])
			Min_Intensity = Min(Min_Intensity, Image[x,y])
	
	return Histogram, Min_Intensity, Max_Intensity

def New_Pixel_Value(Current_Value, Min, Max):
	Target_Max = 255
	Target_Min = 0
	return (Target_Min + (Current_Value - Min) * int(Target_Max-Target_Min)/(Max-Min))

def Histogram_Equalization(Image, Min, Max):
	Image_Height = Image.shape[0]
	Image_Width = Image.shape[1]
	Size = (Image_Height, Image_Width)
	
	New_Image = np.zeros(Size, np.uint8)
	
	for x in range(0, Image_Height):
		for y in range(0, Image_Width):
			New_Image[x,y] = New_Pixel_Value(Image[x,y], Min, Max)
	
	return New_Image

def Plot_Eq(Histogram_Equalization):
	plt.figure()
	plt.title(" Histogram Equalization")
	plt.xlabel("Intensity Level")
	plt.ylabel("Intensity Frequency")
	plt.xlim([0, 256])
	
    
	plt.plot(Histogram_Equalization)
	plt.savefig("Histogram Equalization.jpg")
    
def Plot_Histogram(Histogram):
	plt.figure()
	plt.title("GrayScale Histogram")
	plt.xlabel("Intensity Level")
	plt.ylabel("Intensity Frequency")
	plt.xlim([0, 256])
	plt.plot(Histogram)
	plt.savefig("Histogram.jpg")
    
def cumulative_histogram(Histogram):
    cum_hist = Histogram.copy()
    
    for i in np.arange(1, 256):
        cum_hist[i] = cum_hist[i-1] + cum_hist[i]
        
    return [sum(Histogram[:i+1]) for i in range(len(Histogram))]   
def get_histogram(Image):
  '''
  calculate the normalized histogram of an image
  '''
  height, width = Image.shape
  Histogram = [0.0] * 256
  for i in range(height):
    for j in range(width):
      Histogram[Image[i, j]]+=1
  return np.array(Histogram)/(height*width) 
def normalize_histogram(Image):
  # calculate the image histogram
  Hist= get_histogram(Image)
  # get the cumulative distribution function
  cdf = np.array(cumulative_histogram(Hist))
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize the normalization values
  height, width = Image.shape
  Y = np.zeros_like(Image)
  for i in range(0, height):
    for j in range(0, width):
      Y[i, j] = sk[Image[i, j]]
  # optionally, get the new histogram for comparison
 # new_hist = get_histogram(Y)
  # return the transformed image
  return Y




 

def getRed(redVal):

    return '#%02x%02x%02x' % (redVal, 0, 0)

 

def getGreen(greenVal):

    return '#%02x%02x%02x' % (0, greenVal, 0)

   

def getBlue(blueVal):

    return '#%02x%02x%02x' % (0, 0, blueVal)

 

 

# Create an Image with specific RGB value
def RGB_IMAGE(image):

    image =Image.open(image)

    image.putpixel((0,1), (1,1,5))
    image.putpixel((0,2), (2,1,5))
    histogram = image.histogram()
 

# Modify the color of two pixels

#image.putpixel((0,1), (1,1,5))

#image.putpixel((0,2), (2,1,5))

 

# Display the image

#image.show()

 

# Get the color histogram of the image

#histogram = image.histogram()

 

# Take only the Red counts

    l1 = histogram[0:256]

 

# Take only the Blue counts

    l2 = histogram[256:512]

 

# Take only the Green counts

    l3 = histogram[512:768]

 

    plt.figure(0)

 

# R histogram

    for i in range(0, 256):
 
        plt.bar(i, l1[i], color = getRed(i), edgecolor=getRed(i), alpha=0.3)
        plt.xlabel('intensity')
        plt.ylabel('intensity freq ')

    plt.savefig('R hist')
#    plt.show()
# G histogram

    plt.figure(1)

    for i in range(0, 256):

         plt.bar(i, l2[i], color = getGreen(i), edgecolor=getGreen(i),alpha=0.3)
         plt.xlabel('intensity')
         plt.ylabel('intensity freq ')
    plt.savefig('g hist')    
#    plt.show()

 

 

# B histogram

    plt.figure(2)

    for i in range(0, 256):

         plt.bar(i, l3[i], color = getBlue(i), edgecolor=getBlue(i),alpha=0.3)
         plt.xlabel('intensity')
         plt.ylabel('intensity freq ')
    plt.savefig('B hist')    
#    plt.show()

 

    plt.show()

# blur and grayscale before thresholding
def global_thresholding(image):
    blur = skimage.color.rgb2gray(image)
    blur = skimage.filters.gaussian(blur, sigma=2)

# perform  thresholding
    mask = blur < 0.8
# use the mask to select the "interesting" part of the image
    sel = np.zeros_like(image)
    sel[mask] = image[mask]
    return sel

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def local_thresholding(image):
    blur = skimage.color.rgb2gray(image)
    blur = skimage.filters.gaussian(blur, sigma=2)
    gray_im=rgb2gray(image)
    gray_im =filters.gaussian_Filter(2,image.shape)##smoothing
    pixel_number = gray_im.shape[0] * gray_im.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray_im, np.arange(0,257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]: # This goes from 1 to 254 uint8 range 
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weigth
        Wf = pcf * mean_weigth

        mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
        #print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = image.copy()
    print(final_thresh)
    final_img[image > final_thresh] = 255
    final_img[image < final_thresh] = 0
    return final_img
