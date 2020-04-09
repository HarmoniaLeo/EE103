import numpy as np
import cv2

img = cv2.imread("base_iro.jpg")
img1=img[...,0]/3+img[...,1]/3+img[...,2]/3
img2=img[...,0]*0.299+img[...,1]*0.587+img[...,2]*0.114
cv2.imwrite('way1-2.jpg', np.concatenate((img1,img2),1))
