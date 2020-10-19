from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QComboBox , QMessageBox, QAction, QLineEdit
from PyQt5.QtGui import QPixmap, QIcon,QImage,qRgb
from PyQt5.QtCore import pyqtSlot
from GUI import Ui_MainWindow
import CV404Filters as filters
import CV404Frequency as Hybrid
import CV404Histograms as histograms
import CV404ActiveContour as contour
import CV404Harris as Harris
import sys
import time
import math
import cv2
from scipy.signal import convolve2d
#import pyqtgraph as pg
import numpy as np
from PIL.ImageQt import ImageQt
from PIL import Image
from scipy import ndimage, signal
import CV404Hough as hough
import threading
import qimage2ndarray

from CV404SIFT import SIFT
import CV404Template as TM
import matplotlib.pyplot as plt
import nms
import timeit

class ApplicationWindow (QtWidgets.QMainWindow): 
         
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_filters_load.clicked.connect (self.browes_Filters)
        self.ui.comboBox.currentTextChanged.connect(self.SelectFilter)
        self.ui.comboBox_2.currentTextChanged.connect(self.SelectNoise)
        self.selectNoisee=False
        self.ui.label_histograms_output_3.mousePressEvent =self.getpos
        self.ui.pushButton_histograms_load.clicked.connect (self.browes_histogram)
        
        self.ui.pushButton_histograms_load_2.clicked.connect (self.browes_hybrid_1)
        self.ui.pushButton_histograms_load_3.clicked.connect (self.browes_hybrid_2)
        self.ui.pushButton_histograms_load_4.clicked.connect (self.hybrid)
        self.ui.pushButton.clicked.connect (self.equalization)
        self.ui.pushButton_2.clicked.connect (self.normalization)
        self.ui.pushButton_3.clicked.connect (self.L_thresholding) 
        self.ui.pushButton_4.clicked.connect (self.G_thresholding)
        self.ui.comboBox_3.currentTextChanged.connect(self.RGB)        
        ###################################### task2 ############################################################
        ## Harris
        self.ui.pushButton_harris_load.clicked.connect (self.browes_Harris)
        self.ui.comboBox_6.currentTextChanged.connect(self.Select_Corner)                      
        ## Snake
        self.ui.pushButton_snake_load.clicked.connect (self.browes_Activecontor)
        self.ui.Apply_Harris_3.clicked.connect (self.apply)
        self.ui.Apply_Harris_4.clicked.connect (self.reset)
        self.ui.Apply_Harris_2.clicked.connect (self.clear)
        self.ui.comboBox_4.currentTextChanged.connect(self.option_Filter)
        self.ui.comboBox_5.currentTextChanged.connect(self.option_gradiant)
        self.ui.pushButton_5.clicked.connect (self.chaincode)        
        ## Hough
        self.ui.Apply_Hough.clicked.connect(self.Apply)
        self.ui.pushButton_hough_load.clicked.connect(self.browes_Hough)
        
        global filterofcontour
        filterofcontour=" "
        global gradiantofcontour
        gradiantofcontour=" "
        self.count=0               
        ############################################### task3 ##################################################
        ## TemplateMatching
        self.ui.pushButton_TM_load_A.clicked.connect (self.browes_TM_A)
        self.ui.pushButton_TM_load_B.clicked.connect (self.browes_TM_B)
        self.ui.comboBox_7.currentTextChanged.connect(self.option_Matching)
        self.ui.pushButton_TM_match.clicked.connect (self.Match)
        ## SIFT
        self.ui.pushButton_sift_load_A.clicked.connect (self.browes_SIFT_A)
        self.ui.pushButton_sift_load_B.clicked.connect (self.browes_SIFT_B)
        self.ui.pushButton_sift_match.clicked.connect (self.MatchSIFT)
                                   
    def browes_Filters(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_filters_input.setPixmap(self.pixmap)
        self.ui.label_filters_input.setScaledContents(True)
        self.ui.label_filters_input.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label.setText(self.string)
        self.ui.label_2.setText(str(self.size))
        
    def browes_histogram(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_input.setPixmap(self.pixmap)
        self.ui.label_histograms_input.setScaledContents(True)
        self.ui.label_histograms_input.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_10.setText(str(self.size))
        self.ui.label_11.setText(self.string)
        Histogram_GrayScale, Min, Max = histograms.Histogram_Computation(self.img)
        Histogram_Gray = histograms.Histogram_Computation(self.img)
        for i in range(0,len(Histogram_Gray)):
            print("Histogram[",i,"]: ", Histogram_Gray[i])
        histograms.Plot_Histogram(Histogram_GrayScale)
        himage=cv2.imread("Histogram.jpg", cv2.IMREAD_GRAYSCALE)
        Img=np.asarray(himage, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_hinput.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_hinput.setScaledContents(True)
        self.ui.label_histograms_hinput.setAlignment(QtCore.Qt.AlignCenter)
        histograms.RGB_IMAGE(self.f)
            
    def browes_hybrid_1(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_input_2.setPixmap(self.pixmap)
        self.ui.label_histograms_input_2.setScaledContents(True)
        self.ui.label_histograms_input_2.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_13.setText(str(self.size))
        self.ui.label_12.setText(self.string)
        self.img1=self.img
          
    def browes_hybrid_2(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_hinput_2.setPixmap(self.pixmap)
        self.ui.label_histograms_hinput_2.setScaledContents(True)
        self.ui.label_histograms_hinput_2.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_14.setText(str(self.size))
        self.ui.label_15.setText(self.string)
        self.img2=self.img
                       
    def hybrid(self):
        row1, col1 = self.img1.shape
        row2, col2 = self.img2.shape
        if (self.img1.shape < self.img2.shape):
           img22=cv2.resize(self.img2,(int(col1),int(row1)))
           img11=self.img1
        else:
           img11=cv2.resize(self.img1, (int(col2),int(row2)))
           img22=self.img2
        Hybrid.HPF(img11)
        Hybrid.LPF(img22)
        hybrid_img=Hybrid.Hybrid()
        Img=np.asarray(hybrid_img, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_output_2.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_output_2.setScaledContents(True)
        self.ui.label_histograms_output_2.setAlignment(QtCore.Qt.AlignCenter)
                       
    def browes(self):
        fileName, _filter  = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter (*.jpg *.png *.jpeg *.bmp)")
        if fileName:
            self.f=fileName
            self.pixmap = QPixmap(fileName)
            self.img=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) 
            self.size=self.img.shape
#            print(self.img.shape)
            string=fileName
            while(string.count('/') !=0):
                index=string.find('/')
                index=index+1
                string=string[index:]
                print (string)
                print (string.count('/'))
            indexf=string.find('.')
            self.string=string[:indexf]
#            print (string)
#            self.img=cv2.imread(fileName)
                                      
    def SelectFilter(self, text):
#        print("ok")
        if self.selectNoisee==True:
            input_Img =self.Noisy
            self.ui.label_filters_input.setPixmap(QPixmap.fromImage(self.Nimage))
        else:
            input_Img=self.img
        if text=="Average filters":
           FilterImg=filters.averageFilter(input_Img)
        elif text=="Gaussian filter":
           FilterImg=convolve2d(input_Img, filters.gaussian_Filter(0.3,(3,3)),mode='same')
        elif text=="median filter":  
           FilterImg=filters.median_filter(input_Img,3)
        elif text=="Roberts edge detector":
           FilterImg=filters.roberts_cross(input_Img )
        elif text=="Prewitt edge detector":
           FilterImg=filters.Prewitt( input_Img)
        elif text=="Sobel edge detector":  
           FilterImg , G , theta =filters.sobel( input_Img )
        elif text=="canny edge detector":
           FilterImg=filters.Canny(input_Img ,5 ,40 )       
        elif text=="":
            FilterImg =self.img
        
        Img=np.asarray(FilterImg, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_filters_output.setPixmap(QPixmap.fromImage(image))
        self.ui.label_filters_output.setScaledContents(True)
        self.ui.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)
        time.sleep(0.5)
                              
    def SelectNoise(self, text):
        if text=="Uniform noise":
            image_Noise=filters.im_w_uniform_noise(0,100,self.img)
        elif text=="Gaussian noise":
            image_Noise = filters.im_gaussian_noise(0, 10, self.img )
        elif text=="Salt & pepper noise":
            image_Noise = filters.salt_pepper_noise(self.img, 0.1)
        elif text=="":
            image_Noise =self.img
        self.Noisy=image_Noise
        self.selectNoisee = True
        Img=np.asarray(image_Noise, dtype=np.uint8)
        Noise_image = Image.fromarray(Img)
        self.Nimage = ImageQt(Noise_image)
        self.ui.label_filters_output.setPixmap(QPixmap.fromImage(self.Nimage))
        self.ui.label_filters_output.setScaledContents(True)
        self.ui.label_filters_output.setAlignment(QtCore.Qt.AlignCenter)
        time.sleep(0.5)
        
    def equalization(self):
        Histogram_GrayScale, Min, Max = histograms.Histogram_Computation(self.img)
        New_Image = histograms.Histogram_Equalization(self.img, Min, Max)
        Img=np.asarray(New_Image, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_output.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_output.setScaledContents(True)
        self.ui.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)
#        Histogram_GrayScale2, Min, Max = histograms.Histogram_Computation(New_Image)
#        Histogram_Gray = histograms.Histogram_Computation(New_Image)
        for i in range(0,len(New_Image)):
            print("Histogram_Equalization[",i,"]: ", New_Image[i])
        histograms.Plot_Eq(New_Image)
#        histograms.Plot_Histogram2(Histogram_GrayScale2)
        himage=cv2.imread("Histogram Equalization.jpg", cv2.IMREAD_GRAYSCALE)
        Img=np.asarray(himage, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_houtput.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_houtput.setScaledContents(True)
        self.ui.label_histograms_houtput.setAlignment(QtCore.Qt.AlignCenter)
        time.sleep(0.5)
        
    def normalization(self):
        New_Image = histograms.normalize_histogram(self.img)
        Img=np.asarray(New_Image, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_output.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_output.setScaledContents(True)
        self.ui.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)
#        Histogram_GrayScale2, Min, Max = histograms.Histogram_Computation(New_Image)
#        Histogram_Gray = histograms.Histogram_Computation(New_Image)       
        time.sleep(0.5)
        
    def G_thresholding(self): 
        New_Image=histograms.global_thresholding(self.img)
        Img=np.asarray(New_Image, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_output.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_output.setScaledContents(True)
        self.ui.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)
#        Histogram_GrayScale2, Min, Max = histograms.Histogram_Computation(New_Image)
#        Histogram_Gray = histograms.Histogram_Computation(New_Image)       
    def L_thresholding(self):
        New_Image =histograms.local_thresholding(self.img)
        Img=np.asarray(New_Image, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_output.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_output.setScaledContents(True)
        self.ui.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)
#        Histogram_GrayScale2, Min, Max = histograms.Histogram_Computation(New_Image)
#        Histogram_Gray = histograms.Histogram_Computation(New_Image)        
    def RGB(self, text):      
        if text=="RED":
           Img=cv2.imread('R hist.png')
#           Img=histograms.averageFilter(self.img)
        elif text=="BLUE":
            Img=cv2.imread('B hist.png')
#            Img=histograms.averageFilter(self.img)
        elif text=="GREEN":
            Img=cv2.imread('g hist.png')
            
        Img=np.asarray(Img, dtype=np.uint8)
        Filter_image = Image.fromarray(Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_output.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_output.setScaledContents(True)
        self.ui.label_histograms_output.setAlignment(QtCore.Qt.AlignCenter)
        time.sleep(0.5)
                
########################################## Task2 ##################################
    def browes_Harris (self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_filters_input_5.setPixmap(self.pixmap)
        self.ui.label_filters_input_5.setScaledContents(True)
        self.ui.label_filters_input_5.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_39.setText(self.string)
        self.ui.label_34.setText(str(self.size))
        
    def Select_Corner(self, text):
        if text=="Thresholding":
           threshold = self.ui.textEdit_5.toPlainText()
           if  not threshold:
              QMessageBox.about(self, "Error", "Enter threshold value")
           else:       
              corners_th = Harris.Harris_corner_detector(self.img)
              corners = Harris.Thresholding(corners_th,float(threshold)) 
        if text=="Local thresholding":
          r,ixx,iyy,ixy = Harris.Harris_corner_detector(self.img)
          corners = Harris.Local_Thresholding(ixx,iyy,ixy)
        if text=="Non_maxima supression":
          corners_maxima=Harris.Harris_corner_detector(self.img)
          corners = Harris.Non_maximum_supression(corners_maxima)
    
        Img=np.asarray(corners, dtype=np.uint8)
        corners_image = Image.fromarray(Img)
        self.Nimage = ImageQt(corners_image)
        self.ui.label_filters_output_5.setPixmap(QPixmap.fromImage(self.Nimage))
        self.ui.label_filters_output_5.setScaledContents(True)
        self.ui.label_filters_output_5.setAlignment(QtCore.Qt.AlignCenter)
        time.sleep(0.5)  
        
    def browes_Activecontor(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_output_3.setPixmap(self.pixmap)
        self.ui.label_histograms_output_3.setScaledContents(True)
        self.ui.label_histograms_output_3.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_40.setText(self.string)
        self.ui.label_35.setText(str(self.size))
        self.init_x=[]
        self.init_y=[]
        self.img_display=cv2.imread(self.f, cv2.IMREAD_GRAYSCALE)
        QMessageBox.about(self, "Note", "Please put excpected iteration. Sometimes; if in diffrent alpha, beta and gama, program enters infinite loop and stops responding")
        QMessageBox.about(self, "Note", "Use mouse to put points around the shape (put any number) ")
        
    def apply(self):
        global filterofcontour
        global gradiantofcontour
            
        alpha = self.ui.textEdit.toPlainText()
        beta = self.ui.textEdit_2.toPlainText()
        gama = self.ui.textEdit_3.toPlainText()
        thershold=self.ui.textEdit_4.toPlainText()
        no_iteration=self.ui.textEdit_7.toPlainText()
                       
        if(filterofcontour==" "):
             QMessageBox.about(self, "Error","Choose filter and gradiant first")
        elif(filterofcontour=="Gaussian"):
            FilterImg=convolve2d(self.img, filters.gaussian_Filter(0.3,(3,3)),mode='same')
        elif(filterofcontour=="Average"):
             FilterImg=filters.averageFilter(self.img)
        elif(filterofcontour=="Median"):
            FilterImg=filters.median_filter(self.img,3)
        print(filterofcontour)
        
        if(gradiantofcontour==" "):
             QMessageBox.about(self, "Error", "Choose filter and gradiant first")
        elif(gradiantofcontour=="sobel"):
                img_gradX=ndimage.sobel(FilterImg,axis=0)
                img_gradY=ndimage.sobel(FilterImg,axis=1)
                img_grad=np.hypot(img_gradX,img_gradY)
        elif(gradiantofcontour=="high pass"):
            img_grad=Hybrid.HPF(self.img)
        elif(gradiantofcontour=="Prewitt edge detector"):
            img_grad=filters.Prewitt( self.img)   
        elif(gradiantofcontour=="canny edge detector"):
            img_grad=filters.Canny(self.img ,5 ,40 ) 
                 
        try:
            new_contourx ,new_contoury=contour.Snake(img_grad ,self.init_x,self.init_y, alpha , beta, gama ,thershold )
        except:
            QMessageBox.about(self, "Error", "Make sure you assign alpha, beta, gama and thershold of curvature")
            
        change=len(self.init_x)
        iteration=0
        prex=[]
        prey=[]
        prex.append(self.init_x)
        prey.append(self.init_y)
        self.newX=[]
        self.newY=[]
        
        movepoint=0.3*len(self.init_x)                   
        print(change)
        print(movepoint)
        while(change>movepoint and iteration<int(no_iteration) ):
            print("iteration")
            iteration=iteration+1
            print(iteration)
            self.newX=[]
            self.newY=[]
            change=0
            # draw in all iteration 
#            for i in range (len(new_contourx)-2):
#                frame=cv2.line(self.img_display , (int(new_contourx[i]),int(new_contoury[i])),(int(new_contourx[i+1]),int(new_contoury[i+1])),(255,0,255), 3)
#                time.sleep(0.5)
#            image=Image.fromarray(frame)
#            img= ImageQt(image)
#            self.ui.label_histograms_output_3.setPixmap(QPixmap.fromImage(img))
#            self.ui.label_histograms_output_3.setAlignment(QtCore.Qt.AlignCenter)
#            time.sleep(0.5)
            
            for j in range (len(new_contourx)-2):
                self.newX.append(new_contourx[j+1])
                self.newY.append(new_contoury[j+1])
                if((new_contourx[j]!=prex[-1][j]) or ( new_contoury[j]!=prey[-1][j])):
                    change+=1                   
#            print(senewX)
#            print(newY)            
            prex.append(self.newX)
            prey.append(self.newY)
            try:
                new_contourx ,new_contoury=contour.Snake(img_grad ,self.newX,self.newY, alpha , beta, gama ,thershold )
            except:
                continue                
            time.sleep(1)
            print(change)
            
        self.img_final=cv2.imread(self.f, cv2.IMREAD_GRAYSCALE)
        for i in range (len(self.newX)):
            frame=cv2.rectangle(self.img_display , (int(self.newX[i]),int(self.newY[i])),(int(self.newX[i]+1),int(self.newY[i]+1)),(0,255,255), 1)
            time.sleep(0.5)
            image=Image.fromarray(frame)
            img= ImageQt(image)
            self.ui.label_histograms_output_3.setPixmap(QPixmap.fromImage(img))
            self.ui.label_histograms_output_3.setAlignment(QtCore.Qt.AlignCenter)
            time.sleep(0.5)
        print("end")
#                               
    def reset(self):
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_output_3.setPixmap(self.pixmap)
        self.ui.label_histograms_output_3.setScaledContents(True)
        self.ui.label_histograms_output_3.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_40.setText(self.string)
        self.ui.label_35.setText(str(self.size))
        self.init_x.clear()
        self.init_y.clear()
        self.count=0
        self.img_display=cv2.imread(self.f, cv2.IMREAD_GRAYSCALE)
        
    def clear(self):
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_output_3.setPixmap(self.pixmap)
        self.ui.label_histograms_output_3.setScaledContents(True)
        self.ui.label_histograms_output_3.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_40.setText(self.string)
        self.ui.label_35.setText(str(self.size))
        
    def getpos (self , event):
         global x
         global y        
         self.x=math.floor((event.pos().x()*self.size[0])/self.ui.label_histograms_output_3.frameGeometry().width())
#         print (self.x)
         self.y=math.floor((event.pos().y()*self.size[1]) /self.ui.label_histograms_output_3.frameGeometry().height())
#         print (self.y)
         self.count=self.count+1
         self.init_x.append(self.x)
         self.init_y.append(self.y)
         frame=cv2.rectangle(self.img_display , (self.x,self.y),(self.x+1,self.y+1),(0,255,0), 1)
         image=Image.fromarray(frame)
         img= ImageQt(image)
         self.ui.label_histograms_output_3.setPixmap(QPixmap.fromImage(img))
         self.ui.label_histograms_output_3.setAlignment(QtCore.Qt.AlignCenter)
         print("arr")
         print(self.init_x)
         print(self.init_y)
         
    def option_Filter(self, text):
        global filterofcontour
        filterofcontour=text

    def option_gradiant(self, text):
        global gradiantofcontour
        gradiantofcontour=text
        
    def chaincode(self):

        STRING=" "
        for i in range(len(self.newX)-2):
            ListOfPoints = contour.generates_points_of_line(int(self.newX[i]), int(self.newY[i]),int(self.newX[i+1]), int(self.newY[i+1])) 
            chainCode = contour.generateChainCode(ListOfPoints) 
            chainCodeString = "".join(str(e) for e in chainCode) 
            STRING=STRING+chainCodeString           
#            print(STRING)
#            print (chainCodeString) 
        print("chain Code")
        print(STRING)
        QMessageBox.about(self, "Chain Code"," Printing in python console because it's too long" )
      
######################Hough####################################################            
    def browes_Hough(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_filters_input_4.setPixmap(self.pixmap)
        self.ui.label_filters_input_4.setScaledContents(True)
        self.ui.label_filters_input_4.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_36.setText(self.string)
        self.ui.label_31.setText(str(self.size))
       
    def line(self):
        threshold = self.ui.textEdit_6.toPlainText()
        if  not threshold:
               QMessageBox.about(self, "Error", " You should enter a threshold (2 - 60)")
        else:
            input_img =cv2.imread(self.f)
            gray_img =cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
            img_blurred =cv2.GaussianBlur(gray_img, (5, 5), 1.5)
            edges=cv2.Canny(img_blurred, 100, 200)
            H, rhos, thetas = hough.hough_lines_acc(edges)
            N_Lines =int(self.ui.textEdit_6.toPlainText())#select threshold from gui
            indicies, H =hough.peaks_line(H, N_Lines, nhood_size=12) # find peaks
        #hough.plot_hough_acc(H) # plot hough space, brighter spots have higher votes
            h=hough.hough_lines_draw(input_img, indicies, rhos, thetas)
        #Img=np.asarray(input_img, dtype=np.uint8)
            #image = Image.fromarray(input_img)
            #new_image = image.convert("L")
            Result =qimage2ndarray.array2qimage(input_img)
            #Result= ImageQt(new_image)
            self.ui.label_filters_output_4.setPixmap(QPixmap.fromImage(Result))
            self.ui.label_filters_output_4.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.label_filters_output_4.setScaledContents(True)
                
    def circle(self):
       img = cv2.imread(self.f, cv2.IMREAD_COLOR)
       input_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale image
       img_smooth =cv2.erode(input_img, np.ones((3,)*2,np.uint8), 1)
       img_blur=cv2.GaussianBlur(img_smooth, (3,)*2, 2)
       edge_img=cv2.Canny(img_blur, 40, 80)
       centers, radii = hough.find_circles(edge_img, [20, 40], threshold=110, nhood_size=50)
       img_circles = img.copy()
       for i in range(len(radii)):
            img_circles = hough.hough_circles_draw(img_circles, 'output/ps1-8-a-1.png',
                                         centers[i], radii[i])
       Result =qimage2ndarray.array2qimage(img_circles)
       #Result= ImageQt(new_image)
       self.ui.label_filters_output_4.setPixmap(QPixmap.fromImage(Result))
       self.ui.label_filters_output_4.setAlignment(QtCore.Qt.AlignCenter)
       self.ui.label_filters_output_4.setScaledContents(True)
        
    def Apply(self):
        
        if self.ui.checkBox.isChecked():
            self.line()
        elif self.ui.checkBox_3.isChecked():
             self.circle()    
        else:
           exit()                        
########################################### task 3 #################################################### 
    ### Template Matching 
    def browes_TM_A(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_input_3.setPixmap(self.pixmap)
        self.ui.label_histograms_input_3.setScaledContents(True)
        self.ui.label_histograms_input_3.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_16.setText(self.string)
        self.ui.label_19.setText(str(self.size))
        self.img1= cv2.imread(self.f)
        img_rgb=np.array(self.img1)
        self.img_gray=TM.rgb2gray(img_rgb)
                       
    def browes_TM_B(self):
        self.browes()
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_hinput_3.setPixmap(self.pixmap)
        self.ui.label_histograms_hinput_3.setScaledContents(True)
        self.ui.label_histograms_hinput_3.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_17.setText(self.string)
        self.ui.label_18.setText(str(self.size))
        img_crop= cv2.imread(self.f)
        img_rgb_crop=np.array(img_crop)
        self.img_gray_crop=TM.rgb2gray(img_rgb_crop)
        QMessageBox.about(self, "Note", "when you select the method in comboBox the GUI can be not responce when calculate coorelation ")
        
    def option_Matching(self,text):
        type=text
        self.start=timeit.default_timer()
        if type=='Correlation':
            self.result_TM = TM.match_template_corr(self.img_gray ,self.img_gray_crop)
            self.w=0.0001            
        elif type=='Zero-mean Correlation':
            self.result_TM= TM .match_template_corr_zmean(self.img_gray ,self.img_gray_crop)
            self.w=0.0001
        elif type=='Sum of Squared Differences (SSD)':
            self.result_TM = TM.match_template_ssd(self.img_gray ,self.img_gray_crop)
            self.w=0.05
        elif type=='Normalized Cross Correlation':
            self.result_TM = TM.match_template_xcorr(self.img_gray ,self.img_gray_crop)
            self.w=0.0001
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.result_TM, aspect='auto')
        fig.savefig('output')
        time.sleep(0.5)
        self.end1=timeit.default_timer()
        QMessageBox.about(self, "Note", "thershold must be integer")
        QMessageBox.about(self, "Note", "by practise  we use theshold for Corr and Corr_mean=380 and for SSD and X_corr =25 ")
        
        
    def Match(self):
        thr = self.ui.textEdit_8.toPlainText()
        try :
           threshold=int(thr)
          
        except:
            QMessageBox.about(self, "Error", "Make sure you assign thershold by integer number")
        
        output_Img=np.asarray(cv2.imread('output.png', cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
        Filter_image = Image.fromarray(output_Img)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_input_4.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_input_4.setScaledContents(True)
        self.ui.label_histograms_input_4.setAlignment(QtCore.Qt.AlignCenter)
        time.sleep(0.5)
        try :
           Result =TM.local_maxima(self.result_TM,threshold,self.w)
           print(Result)           
        except:
            QMessageBox.about(self, "Error", "Make sure you assign thershold by integer number")
        w, h = self.img_gray_crop.shape[::-1] 
        img= self.img1
        TM.draw(output_Img,self.img_gray_crop,Result,img)
        
#        output_Img=np.asarray(cv2.imread('output_edit.png'), dtype=np.uint8)
#        Filter_image = Image.fromarray(output_Img)
#        image = ImageQt(Filter_image)
#        self.ui.label_histograms_input_4.setPixmap(QPixmap.fromImage(image))
#        self.ui.label_histograms_input_4.setScaledContents(True)
#        self.ui.label_histograms_input_4.setAlignment(QtCore.Qt.AlignCenter)
#        time.sleep(0.5)
        output_ImgDraw=np.asarray(cv2.imread('output_draw.png'), dtype=np.uint8)
        Filter_image = Image.fromarray(output_ImgDraw)
        image = ImageQt(Filter_image)
        self.ui.label_histograms_hinput_4.setPixmap(QPixmap.fromImage(image))
        self.ui.label_histograms_hinput_4.setScaledContents(True)
        end=timeit.default_timer()
#        print(self.start)
#        print(end)
#        print(end-self.start)
        time1=(end-self.end1)+(self.end1-self.start)
        self.ui.TM_TimeElapsed.setText(str(int(time1))+" sec")
        
    ### SIFT       
    def browessift(self):
        fileName, _filter = QFileDialog.getOpenFileName(self, "Title", "Default File", "Filter (*.jpg *.png *.jpeg *.bmp)")
        if fileName:
            self.f=fileName
            self.pixmap = QPixmap(fileName)
            self.img=np.array(Image.open(fileName))
            self.size=self.img.shape
            string=fileName
            while(string.count('/') !=0):
                index=string.find('/'); index=index+1; string=string[index:]; indexf=string.find('.')
            self.string=string[:indexf]        
    
    def browes_SIFT_A(self):
        self.browessift()
        self.input_image = self.img
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()), QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_input_5.setPixmap(self.pixmap)
        self.ui.label_histograms_input_5.setScaledContents(True)
        self.ui.label_histograms_input_5.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_23.setText(self.string)
        self.ui.label_24.setText(str(self.size))      
        
    def browes_SIFT_B(self):
        self.browessift()
        self.pattern = self.img
        self.pixmap = self.pixmap.scaled(int(self.pixmap.height()), int(self.pixmap.width()),QtCore.Qt.KeepAspectRatio)
        self.ui.label_histograms_hinput_5.setPixmap(self.pixmap)
        self.ui.label_histograms_hinput_5.setScaledContents(True)
        self.ui.label_histograms_hinput_5.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.label_21.setText(self.string)
        self.ui.label_22.setText(str(self.size))
        
    def MatchSIFT(self):
        Sigma = self.ui.textEdit_10.toPlainText()
        if  not Sigma:
           QMessageBox.about(self, "Error", " Insert Sigma value (Recommended: 1.6)")
        else:
           sift = SIFT()
           QMessageBox.about(self, "Warning"," This might take a while, be patient.")
           begin=timeit.default_timer()
           match= sift.start(float(Sigma),self.input_image,self.pattern)
           sift_image = Image.fromarray(match)
           self.Nimage = ImageQt(sift_image)
           self.ui.label_histograms_output_4.setPixmap(QPixmap.fromImage(self.Nimage))
           self.ui.label_histograms_output_4.setScaledContents(True)
           self.ui.label_histograms_output_4.setAlignment(QtCore.Qt.AlignCenter)
           time.sleep(0.5)
           stop =timeit.default_timer()
           print(begin)
           print(stop)
           print(stop-begin)
           self.ui.TM_TimeElapsed_2.setText(str(int(stop-begin))+" sec")
           
                         

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()