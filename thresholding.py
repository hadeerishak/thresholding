import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
fpaths = []
state = 'malignant'
fpaths=glob.glob("C:\\Users\\Owner\\hadeer\\"+state+"\\*.jpg")#you need two slashes after a slash 
for i in range(10):
    #initialize
    filename = fpaths[i]
    temp = fpaths[i]
    length = len(temp)
    pos = temp.rindex("\\")
    fname = temp[pos+1:length]
    pos = fname.find(".")
    fname = fname[0:pos]
    
    img= cv2.imread(fpaths[i],0)
    #morphologic operaration of closing 
    kernel = np.ones((10,10),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #global thresholding (skip the first thing and i want the second thing use (_)bimodal image (In simple words, bimodal image is an image whose histogram has two peaks))
    _,th1 = cv2.threshold(opening,127,255,cv2.THRESH_BINARY)
    #dynamic thresholding (adaptive thresholding, This means that threshold value for binarizing image is not fixed but it is dynamic.)
    th3 = cv2.adaptiveThreshold(th1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    
    
    
    #dynamic thresholding
   # block_size=2777
   # constant=15
    #th2=cv2.adaptiveThreshold(img,500,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size, constant)
   
    #retval, threshold = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.float32)/25
   # dst = cv2.filter2D(th2,-1,kernel)
    #temp =cv2.medianBlur(th2,9)
    cv2.imwrite("a"+fname+"_seg.jpg",th3)
    
