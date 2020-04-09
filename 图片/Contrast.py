import cv2
import numpy as np

A=[0,0.4,0.8,1.2,1.6,2]
imgs=[]

img = cv2.imread("base.jpg",cv2.IMREAD_GRAYSCALE)
img=img.astype(np.int_)
for a in A:
    print((1-a)*np.mean(np.mean(img)))
    img1=a*img+(1-a)*np.mean(np.mean(img))*np.ones(img.shape)
    np.where(img1>255,255,img1)
    imgs.append(img1)
cv2.imwrite("cst.jpg",np.concatenate((np.concatenate(imgs[:3],1),np.concatenate(imgs[3:],1)),0))