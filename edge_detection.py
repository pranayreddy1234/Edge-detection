#!/usr/bin/env python
# coding: utf-8

# In[863]:


import cv2
import numpy as np

img = cv2.imread('Hough.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = cv2.bitwise_not(gray)
bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,                             cv2.THRESH_BINARY, 15, -2)

vertical = np.copy(bw)

rows = vertical.shape[0]
verticalsize = rows // 30
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
vertical = cv2.erode(vertical, verticalStructure)

vertical = cv2.bitwise_not(vertical)

edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C,                                 cv2.THRESH_BINARY, 3, -2)


smooth = np.copy(vertical)
smooth = cv2.blur(smooth, (2, 2))
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]


vertical = np.where(vertical > 192 , 0 , 255)
vertical = np.uint8(vertical)

lines = cv2.HoughLines(vertical,1,np.pi/60,200)

rho__list = []
theta__list = []

for i in range(lines.shape[0]):
    for rho,theta in lines[i]:
        rho__list.append(rho)
        theta__list.append(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imwrite('results/red lines.jpg',img)        
with open('results/red_lines.txt', 'w') as f:
    for i in range(len(rho__list)):
        x = [np.degrees(theta__list[1])-90,-rho__list[i]]
        temp = [90 + abs(x[0]-90),x[1]]
        f.write("%s\n" % temp)        

        
img = cv2.imread('Hough.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img,50,100,apertureSize = 3)
kernel_diagonal = np.zeros((5,5),np.uint8)
np.fill_diagonal(kernel_diagonal, 1)
erosed = cv2.erode(edges,kernel_diagonal,iterations = 1)

kernel = np.ones((5,5),np.uint8)
dilated = cv2.dilate(erosed,kernel,iterations = 4)

kernel = np.zeros((3,3),np.uint8)
np.fill_diagonal(kernel, 1)
erosed = cv2.erode(dilated,kernel,iterations = 20)

lines = cv2.HoughLines(erosed,1,np.pi/180,11)

rho_list = []
theta_list = []

for line in lines:
    rho,theta = line[0]
    if 2.51<theta<2.52 and rho != 257:
        rho_list.append(rho)
        theta_list.append(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
cv2.imwrite('results/blue_lines.jpg',img)        
with open('results/blue_lines.txt', 'w') as f:
    for i in range(len(rho_list)):
        x = [np.degrees(theta_list[1])-90,-rho_list[i]]
        temp = [90 + abs(x[0]-90),x[1]]
        f.write("%s\n" % temp)


        
img = cv2.imread('Hough.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=18, minRadius=29, maxRadius=33)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw inner circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        
cv2.imwrite('results/coins.jpg',img)        
with open('results/coins.txt', 'w') as f:
    for i in range(circles.shape[1]):
        temp = [circles[0][i][1],circles[0][i][0],circles[0][i][2]]
        f.write("%s\n" % temp)


# In[ ]:




