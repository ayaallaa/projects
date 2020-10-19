from os import listdir
from os.path import isfile , join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage, signal, interpolate
from skimage import filters
import math
import sys
import cv2
import CV404Filters as filters

#img=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) 


def circ_replicate(array): #repeat put last as first, and first as last again (circ)
    rows=len(array)
    try:
        columns=len(array[0])
        #print(rows)
        #print(columns)
        arr_replicated=np.tile(array,(3,3)) # copy one more column and one more row
        arr_replicated=arr_replicated[rows-1:2*rows+1, columns-1:2*columns+1]
    except: #1D vectors
        arr_replicated=np.tile(array,3) # copy one more column and one more row
        arr_replicated=arr_replicated[rows-1:2*rows+1] # truncate extraPlus rows 
#    print(arr_replicated)
    return arr_replicated

#print(circ_replicate([[1,2,-3],[0,2,3]])) #[x1, x2, x3], [y1, y2, y3]
    
def get_4neighbors(pointsX, pointsY): #
    pointsOut= np.array([[pointsX, pointsY], [pointsX-1, pointsY], [pointsX+1, pointsY],\
                         [pointsX, pointsY-1], [pointsX, pointsY+1]    ] )
    return pointsOut

def get_8neighbors(pointsX, pointsY): #
    pointsOut= np.array([[pointsX, pointsY],[pointsX-1, pointsY-1], [pointsX-1, pointsY],[pointsX-1, pointsY+1], 
                        [pointsX+1, pointsY-1],[pointsX+1, pointsY],[pointsX+1, pointsY+1],
                         [pointsX, pointsY-1], [pointsX, pointsY+1]] )
    return pointsOut

def normalize(array, newMin, newMax):
    minArr=array.min()
    maxArr=array.max()
    return ((array-minArr)/(maxArr-minArr))*(newMax-newMin)+newMin

def compute_energy(pointsX, pointsY, alpha, beta, gamma, grad_normalized , c_thershold ,mag_thershold,flag): #compute continuity energy
    #print(pointsX)
    #print(pointsY)
    newPointsX=np.zeros(pointsX.shape)
    newPointsY=np.zeros(pointsY.shape)
#    newPointsX2=[]
#    newPointsY2=[]
    grid_x, grid_y= np.mgrid[0:len(grad_normalized), 0:len(grad_normalized[0])]
#    print(grid_x.shape)
    distance=0 #compute average distance 

    
    for ind in range (len(pointsX)-1): #range(len(pointsX)-2)=Equivalent:  
        distance+=np.sqrt((pointsX[ind]-pointsX[ind+1]) ** 2+(pointsY[ind]-pointsY[ind+1]) ** 2)
    distance/=(len(pointsX)-2)
#    print(distance)
    
    for ind in range (len(pointsX)-1): #compute elastic energy VECTOR #cover all Core points
        # having circular previous and circ next
        #print(pointsX[ind])        
        #if False: #block comment
 
        elas_En=np.zeros(9) # For the Current contour point and its 4-neighbors
        curv_En=np.zeros(9) # For the Current contour point and its 4-neighbors
        Grad_En=np.zeros(9) # For the Current contour point and its 4-neighbors
        
        allpoints_8=get_8neighbors(pointsX[ind],pointsY[ind]) # retrieve neighbors
        #print(allpoints_4)
        Grad_En =  ndimage.map_coordinates(grad_normalized, np.transpose(allpoints_8))#spline
        Grad_En_norm =normalize(Grad_En, 0, 1)
        print("Grad_En_norm")
        print(Grad_En_norm)
        
        for neigh in range(len(allpoints_8)): #cover all neighbors           
            elas_En[neigh]=(distance-(np.sqrt((allpoints_8[neigh,0]-pointsX[ind-1]) ** 2+
                                (allpoints_8[neigh,1]-pointsY[ind-1]) ** 2)))#neighbors
            
#            print(elas_En[neigh])
#            elas_En[neigh]=elas_En[neigh]*alpha
            #print(allpoints_4[neigh,0],allpoints_4[neigh,1],pointsX[ind-1],pointsY[ind-1],\
            #     pointsX[ind+1],pointsY[ind+1])
#            print(elas_En[neigh])
            curv_En[neigh]=np.sqrt((2*allpoints_8[neigh,0]-pointsX[ind+1]-pointsX[ind-1]) **2 +
                        (2*allpoints_8[neigh,1]-pointsY[ind+1]-pointsY[ind-1]) **2) 
#            print(curv_En[neigh])
#            print(Grad_En[neigh])
#            curv_En[neigh]= curv_En[neigh]*beta
#        print("energy")
#        print(elas_En)
#        print(curv_En)
#        print(Grad_En)
     
        elas_En=float(alpha)*elas_En
        curv_En=float(beta)*curv_En
        Grad_En=float(gamma)*Grad_En_norm

        total_En=elas_En+curv_En+Grad_En_norm
#        print(total_En)
        indMin=np.argmin(total_En)
#        print(pointsX[ind], pointsY[ind])
        newPointsX[ind]=allpoints_8[indMin,0]
        newPointsY[ind]=allpoints_8[indMin,1]
#        print("new")
#        print(newPointsX[ind], newPointsY[ind])
#        newPointsX2.append(newPointsX[ind])
#        newPointsY2.append(newPointsY[ind])
        
#        print(pointsX[ind], pointsY[ind])
        C=curvature(newPointsX,newPointsY)
        relax=beta_relaxation(C ,grad_normalized,newPointsX,newPointsY, c_thershold ,mag_thershold)
        if(len(relax)!=0 and flag!="f"):
            for i in range(len(relax)):
                compute_energy(pointsX[relax[i]], pointsY[relax[i]], alpha, 0, gamma, grad_normalized , c_thershold ,mag_thershold,"f")
                print("relax")
            
            
    print("new")
    print(newPointsX, newPointsY)
    return (newPointsX, newPointsY)
def curvature(pointsX,pointsY):
    c=[]

    for i in range (len(pointsX)-2):
        ux1=pointsX[i+1]-pointsX[i]
        uy1=pointsY[i+1]-pointsY[i]
        norm1=np.sqrt((pointsX[i+1]-pointsX[i])**2 +(pointsY[i+1]-pointsY[i])**2)
        ux2=pointsX[i+2]-pointsX[i+1]
        uy2=pointsY[i+2]-pointsY[i+1]
        norm2=np.sqrt((pointsX[i+2]-pointsX[i+1])**2 +(pointsY[i+2]-pointsY[i+1])**2)
        UX=(ux1/norm1)-(ux2/norm2)
        UY=(uy1/norm1)-(uy2/norm2)
        c.append(float(np.sqrt(UX**2 +UY**2)))
    print("c")
    print(c)
    return c

def beta_relaxation(C  ,img_grad_norm,pointsX,pointsY, c_thershold ,mag_thershold):
    index=[]
    for i in range (len(C)-2):
        if (C[i+2]>C[i+1] and C[i]>C[i+1] and C[i]>float(c_thershold) and img_grad_norm[pointsX][pointsY]>mag_thershold ):
            index.append(i+1)
#    print("beta")
    return index




        
def Snake(img ,initX,initY, alpha , beta, gama , thershold ):
#    img_gFilter=signal.convolve2d(img, filters.gaussian_Filter(0.3, (3,3)), mode='same')
#    img_gradX=ndimage.sobel(img_gFilter,axis=0)
#    img_gradY=ndimage.sobel(img_gFilter,axis=1)
#    img_grad=np.hypot(img_gradX,img_gradY)
    img_grad_norm=-normalize(img,0,1)
    
    x_rep=circ_replicate(initX) # circular repeat
    y_rep=circ_replicate(initY) # circular repeat
    
    newContour=compute_energy(x_rep,y_rep, alpha , beta, gama ,img_grad_norm, thershold ,0.6 ,"t")
#    compute_energy(pointsX, pointsY, alpha, 0, gamma, grad_normalized , c_thershold ,mag_thershold,"f")
#    newPointsX, newPointsY =compute_energy(x_rep,y_rep, 0.5, 0.8, 0.4,img_grad_norm)
    
    return newContour


codeList =[5, 6, 7, 4, -1, 0, 3, 2, 1] 

def getChainCode(x1, y1, x2, y2): 
    dx = x2 - x1 
    dy = y2 - y1 
    hashKey = 3 * dy + dx + 4
    return codeList[hashKey] 

def generateChainCode(ListOfPoints): 
    chainCode = [] 
    for i in range(len(ListOfPoints) - 1): 
        a = ListOfPoints[i] 
        b = ListOfPoints[i + 1] 
        chainCode.append(getChainCode(a[0], a[1], b[0], b[1])) 
    return chainCode


def generates_points_of_line(x1, y1, x2, y2): 
    ListOfPoints = [] 
    ListOfPoints.append([x1, y1]) 
    xdif = x2 - x1 
    ydif = y2 - y1 
    dx = abs(xdif) 
    dy = abs(ydif) 
    if(xdif > 0): 
        xs = 1
    else: 
        xs = -1
    if (ydif > 0): 
        ys = 1
    else: 
        ys = -1
    if (dx > dy): 
  
        # Driving axis is the X-axis 
        p = 2 * dy - dx 
        while (x1 != x2): 
            x1 += xs 
            if (p >= 0): 
                y1 += ys 
                p -= 2 * dx 
            p += 2 * dy 
            ListOfPoints.append([x1, y1]) 
    else: 
  
        # Driving axis is the Y-axis 
        p = 2 * dx-dy 
        while(y1 != y2): 
            y1 += ys 
            if (p >= 0): 
                x1 += xs 
                p -= 2 * dy 
            p += 2 * dx 
            ListOfPoints.append([x1, y1]) 
    return ListOfPoints

    
    