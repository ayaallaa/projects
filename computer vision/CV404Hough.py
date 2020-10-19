# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

####LINE_DETECTION_USING_HOUGH#################################################

# read in shapes image and convert to grayscale
shapes = cv2.imread('1.jpg')
#cv2.imshow('Original Image', shapes)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)
#shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)

# find Canny Edges and show resulting image
#canny_edges = cv2.Canny(shapes_blurred, 100, 200)
#cv2.imshow('Canny Edges', canny_edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
########################################### 

# This is the function that will build the Hough Accumulator for the given image
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    ''' A function for creating a Hough Accumulator for lines in an image. '''
    height, width = img.shape # 
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


def hough_simple_peaks(H, num_peaks):
    ''' A function that returns the number of indicies = num_peaks of the
        accumulator array H that correspond to local maxima. '''
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    return np.vstack(np.unravel_index(indices, H.shape)).T


def peaks_line(H, num_peaks, threshold=0, nhood_size=3):
    
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x),int( max_x),1):
            for y in range(int(min_y), int(max_y),1):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H


# a simple funciton used to plot a Hough Accumulator
def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    
    fig = plt.figure(figsize=(10, 50))
    fig.canvas.set_window_title(plot_title)
    	
    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()


# drawing the lines from the Hough Accumulatorlines using OpevCV cv2.line
def hough_lines_draw(img, indicies, rhos, thetas):
   
    for i in range(len(indicies)):
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (255,0, 0), 2)

"""
# run hough_lines_accumulator on the shapes canny_edges image
H, rhos, thetas = hough_lines_acc(canny_edges)
indicies, H = hough_peaks(H, 7, nhood_size=11) # find peaks
plot_hough_acc(H) # plot hough space, brighter spots have higher votes
hough_lines_draw(shapes, indicies, rhos, thetas)

# Show image with Hough Transform Lines
cv2.imshow('Major Lines: Manual Hough Transform', shapes)
cv2.waitKey(0)
cv2.imwrite('Image_wWith_Line.jpg',shapes)
cv2.destroyAllWindows()
"""
#Detecting circles using hough transform ######################################
import cv2
import numpy as np

def hough_circles_acc(edge_img, radius):
    accumulator = np.zeros(edge_img.shape, dtype=np.uint8)
    yis, xis = np.nonzero(edge_img) # coordinates of edges
    num_px = len(xis)
    (m,n) = edge_img.shape
    for x,y in zip(xis,yis):
        theta = np.arange(0,360)
        a = (y - radius * np.sin(theta * np.pi / 180)).astype(np.uint)
        b = (x - radius * np.cos(theta * np.pi / 180)).astype(np.uint)
        valid_idxs = np.nonzero((a < m) & (b < n))
        a, b = a[valid_idxs], b[valid_idxs]
        c = np.stack([a,b], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _,idxs,counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs]
        accumulator[uc[:,0], uc[:,1]] += counts.astype(np.uint)
    return accumulator

def hough_circles_draw(img, outfile, peaks, radius):
    for peak in peaks:
        cv2.circle(img, tuple(peak[::-1]), radius, (0,255,0), 2)
    cv2.imwrite(outfile, img)
    return img

def clip(idx):
    return int(max(idx,0))

def hough_peaks(H, numpeaks=1, threshold=100, nhood_size=5):
    peaks = np.zeros((numpeaks,2), dtype=np.uint64)
    temp_H = H.copy()
    temp_H = temp_H.astype(np.float32)
    for i in range(numpeaks):
        _,max_val,_,max_loc = cv2.minMaxLoc(temp_H) # find maximum peak
        if max_val > threshold:
            peaks[i] = max_loc
            (c,r) = max_loc
            t = nhood_size//2.0
            temp_H[clip(r-t):int(r+t+1), clip(c-t):int(c+t+1)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks[:,::-1]

def find_circles(edge_img, radius_range=[1,2], threshold=100, nhood_size=10):
    n = radius_range[1] - radius_range[0]
    H_size = (n,) + edge_img.shape
    H = np.zeros(H_size, dtype=np.uint)
    centers = ()
    radii = np.arange(radius_range[0], radius_range[1])
    valid_radii = np.array([], dtype=np.uint)
    num_circles = 0
    for i in range(len(radii)):
        H[i] = hough_circles_acc(edge_img, radii[i])
        peaks = hough_peaks(H[i], numpeaks=10, threshold=threshold,
                            nhood_size=nhood_size)
        if peaks.shape[0]:
            valid_radii = np.append(valid_radii, radii[i])
            centers = centers + (peaks,)
            for peak in peaks:
                cv2.circle(edge_img, tuple(peak[::-1]), radii[i]+1, (0,0,0), -1)
        #  cv2.imshow('image', edge_img); cv2.waitKey(0); cv2.destroyAllWindows()
        num_circles += peaks.shape[0]
        print('Progress: %d%% - Circles: %d\033[F\r'%(100*i/len(radii), num_circles))
    print('Circles detected: %d          '%(num_circles))
    centers = np.array(centers)
    return centers, valid_radii.astype(np.uint)
"""
img = cv2.imread('coins.jpg', cv2.IMREAD_COLOR)
smoothed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smoothed_img = cv2.erode(smoothed_img, np.ones((3,)*2,np.uint8), 1)
smoothed_img = cv2.GaussianBlur(smoothed_img, (3,)*2, 2)
edge_img = cv2.Canny(smoothed_img, 40, 80)

   

    #  Detect circles
centers, radii = find_circles(edge_img, [20, 40], threshold=110, nhood_size=50)
img_circles = img.copy()
for i in range(len(radii)):
     img_circles = hough_circles_draw(img_circles, 'output/ps1-8-a-1.png',
                                         centers[i], radii[i])
cv2.imshow('Result',img_circles)    
cv2.waitKey(0)
cv2.imwrite('Image_wWith_Line.jpg',img_circles)
cv2.destroyAllWindows()
"""
