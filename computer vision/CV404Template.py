import numpy as np
from scipy.signal import correlate2d
import itk
from itkwidgets import view
import itkwidgets
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import scipy.ndimage.filters as filters
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy import integrate
from scipy import misc
import scipy.misc
from numpy import asarray
from PIL import Image
import itk
from numpy import asarray



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def match_template_corr( x , temp ):
    y = np.empty(x.shape)
    y = correlate2d(x,temp,'same')
    return y

def match_template_corr_zmean( x , temp ):
    return match_template_corr(x , temp - temp.mean())

def match_template_ssd( x , temp ):
    term1 = np.sum( np.square( temp ))
    term2 = -2*correlate2d(x, temp,'same')
    term3 = correlate2d( np.square( x ), np.ones(temp.shape),'same' )
    ssd = np.maximum( term1 + term2 + term3 , 0 )
    return 1 - np.sqrt(ssd)

def match_template_xcorr( f , t ):
    f_c = f - correlate2d( f , np.ones(t.shape)/np.prod(t.shape), 'same') 
    t_c = t - t.mean()
    numerator = correlate2d( f_c , t_c , 'same' )
    d1 = correlate2d( np.square(f_c) , np.ones(t.shape), 'same')
    d2 = np.sum( np.square( t_c ))
    denumerator = np.sqrt( np.maximum( d1 * d2 , 0 )) # to avoid sqrt of negative
    response = np.zeros( f.shape )
    valid = denumerator > np.finfo(np.float32).eps # mask to avoid division by zero
    response[valid] = numerator[valid]/denumerator[valid]
    return response


#######local maxima########

def local_maxima( x ,threshold,w ):
    neighborhood_size=8
   
    data_max = filters.maximum_filter(x, neighborhood_size)
    print(data_max)
    maxima = (x == data_max)
    print(maxima)
    print(x)

    data_min = filters.minimum_filter(x, neighborhood_size)
    diff = ((w*data_max-w* data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(x, labeled, range(1, num_objects+1)))
    #print(xy)
    return xy
def get_rect_on_maximum(y,template):   
        ij = np.unravel_index(np.argmax(y), y.shape)
        x, y = ij[::-1]
        wtemp, htemp = template.shape
        rect = plt.Rectangle((x-wtemp/2, y-(htemp)/2), wtemp, htemp, edgecolor='r', facecolor='none')
        rect_2 = plt.Rectangle((x-(wtemp+8)/2, y-(htemp+20)/2), wtemp, htemp, edgecolor='r', facecolor='none')
        return rect,x,y,rect_2
def make_rects(plt_object,xy,template):
        wtemp, htemp = template.shape
        print(template.shape)
        for ridx in range(xy.shape[0]):
            y,x = xy[ridx]
            r =  plt.Rectangle((x-(wtemp+8)/2, y-(htemp+20)/2), wtemp, htemp, edgecolor='g', facecolor='none')
            plt_object.add_patch(r)
            
def make_circles(plt_object,xy,template):
        htemp, wtemp = template.shape
        for ridx in range(xy.shape[0]):
            y,x = xy[ridx]
            plt_object.plot(x, y, 'o', markeredgecolor='g', markerfacecolor='none', markersize=10)
            
def draw(output_Img,img_gray_crop,Result,img):
    r,x,y,rect_2 = get_rect_on_maximum(output_Img,img_gray_crop)        
#    plt.figure(figsize=(6,6))
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.imshow(output_Img, cmap = 'gray')
    make_circles(ax, Result,img_gray_crop)
    ax.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(output_Img, aspect='auto')
#    ax.set_axis_off()
    fig.savefig('output_edit.png')
#    fig,ax = plt.subplots(1)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.imshow(img, cmap = 'gray')
    make_rects( ax, Result, img_gray_crop )  
    ax.add_patch(rect_2)
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto')
    fig.savefig('output_draw.png')       
    