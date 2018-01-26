import cv2
import numpy as np
import argparse 
import glob
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

#initialize
filename = "ISIC_0009883.jpg"
block_size=17899
constant=15

img = cv2.imread(filename,0) #zero for greyscale -1 for colour 1 for colour
row,col = img.shape
row_s = np.int(row/2)
col_s = np.int(col/2)
small= cv2.resize(img,(col_s,row_s))
cv2.imshow(filename,small)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#blockSize: Size of pixels neighbourhoods used to calculate threshold
th1= cv2.adaptiveThreshold(small,500,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,block_size, constant)
th2=cv2.adaptiveThreshold(small,500,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size, constant)

output =[small, th1,th2]
titles =['original','Mean Adaptive', 'Gaussian Adaptive']
# remove noise
thresh_blur = cv2.GaussianBlur(th1,(3,3),0)
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(th1,-1,kernel)


e=3;
for i in range(e):
    plt.subplot(1,3,i+1)
    plt.imshow(output[i],cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
cv2.imshow("Adaptive Threshold (Mean)",th1) #adaptive threshold mean
cv2.imshow("Adaptive with Gaussian Blur",thresh_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
#
edges = cv2.Canny(th1,60,90)
edges1 = cv2.Canny(thresh_blur,60,90)

rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(edges, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
img = cv2.line(edges,(cols-1,righty),(0,lefty),(0,255,0),2)

cv2.imshow("Canny with Mean",img)
#cv2.imshow("2Convoloution",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imshow("Canny with Gaussian",edges1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




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
