# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:43:26 2018

@author: Owner
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("ISIC_0011110.jpg",0) #zero for greyscale -1 for colour 1 for colour
row,col = img.shape
row_s = np.int(row/2)
col_s = np.int(col/2)
small= cv2.resize(img,(col_s,row_s))
cv2.imshow('ISIC_0011110.jpg',small)
cv2.waitKey(0)
cv2.destroyAllWindows()
#blockSize: Size of pixels neighbourhoods used to calculate threshold
block_size=17899
constant=11
th1= cv2.adaptiveThreshold(small,500,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size, constant)
th2=cv2.adaptiveThreshold(small,500,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size, constant)
#sobel type of edge detection
sobelx = cv2.Sobel(small,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(small,cv2.CV_64F,0,1,ksize=5)  # y
#laplacian
laplacian = cv2.Laplacian(small,cv2.CV_64F)

output =[img, th1,th2, sobelx, sobely,laplacian]
titles =['original','Mean Adaptive', 'Gaussian Adaptive','Sobel X','Sobel Y','laplacian']
# remove noise
img = cv2.GaussianBlur(small,(3,3),0)



e=6;
for i in range(e):
    plt.subplot(1,6,i+1)
    plt.imshow(output[i],cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
cv2.imshow("small",th1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#thM1 = cv2.adaptiveThreshold(small,600,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,3)#15 is the threshold #the noise value
#cv2.imshow("small",thM1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#small= cv2.resize(img,(100,100))
#cv2.waitKey(0)#is a keyboard binding function. Its argument is the time in milliseconds. The function waits for specified milliseconds for any keyboard event. If you press any key in that time, the program continues. If 0 is passed, it waits indefinitely for a key stroke. It can also be set to detect specific key strokes like, if key a is pressed etc which we will discuss below.
#cv2.destroyAllWindows()#simply destroys all the windows we created. If you want to destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name as the argument.




#edges = cv2.Canny(img,400,300) #canny edge detection 2nd argument is minval and 3rd argument is our maxval
#plt.subplot(121),
#cv2.imshow('gray',small)
#cv2.waitKey(0)
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),
#cv2.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#cv2.show()
