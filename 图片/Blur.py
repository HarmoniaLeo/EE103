import cv2
import numpy as np
import math
import time

def gaussian(u, v, sigma):
    pi = 3.1416
    intensity = 1 / (2.0 * pi * sigma * sigma) * math.exp(- 1 / 2.0 * ((u ** 2) + (v ** 2)) / (sigma ** 2))
    return intensity

def gaussiankernel(r, sigma):
    kernel = np.zeros([r, r])
    for i in range(r):
        for j in range(r):
            kernel[i, j] = gaussian(r-i,r-j, sigma)
    kernel /= np.sum(np.sum(kernel))
    return kernel

def mbkernel(agl,r):
    rad=math.radians(agl)
    w=math.ceil(r*np.abs(np.cos(rad)))
    h=math.ceil(r*np.abs(np.sin(rad)))
    if h==0:
        h+=1
    if w==0:
        w+=1
    d=2/(r+r**2)
    kernel=np.zeros((int(h),int(w)))
    for i in range(0,r):
        if np.sin(rad)>=0:
            y=math.floor(h-i*np.sin(rad)-1)
        else:
            y=math.floor(-i*np.sin(rad))
        if np.cos(rad)>=0:
            x=math.floor(i*np.cos(rad))
        else:
            x=math.floor(w+i*np.cos(rad)-1)
        if kernel[y][x]==0:
            kernel[y][x]=(i+1)*d
        else:
            if np.sin(rad)>=0:
                kernel[y+1][x]=(i+1)*d
            else:
                kernel[y-1][x]=(i+1)*d
    return kernel

def conv(base,kernel):
    tar=np.zeros(base.shape)
    for j in range(kernel.shape[0]-1,base.shape[0]):
        for i in range(kernel.shape[1]-1,base.shape[1]):
            s=0
            for k in range(0,kernel.shape[0]):
                for l in range(0,kernel.shape[1]):
                    s+=base[j-k][i-l]*kernel[k][l]
            tar[j][i]=s
    return tar

def convlist(base,kernel,h,w):
    tar=np.zeros((h,w))
    tar=tar.tolist()
    for j in range(kernel.shape[0]-1,h):
        for i in range(kernel.shape[1]-1,w):
            s=0
            for k in range(0,kernel.shape[0]):
                for l in range(0,kernel.shape[1]):
                    s+=base[j-k][i-l]*kernel[k][l]
            tar[j][i]=s
    return tar


r=50
a=135
base = cv2.imread("base.jpg",cv2.IMREAD_GRAYSCALE)
k=mbkernel(a,r)
tic = time.time()
tar=cv2.filter2D(base,-1,kernel=k)
#tar2=conv(base,k)
toc = time.time()
print(toc-tic)
h=base.shape[0]
w=base.shape[1]
base=base.tolist()
tic = time.time()
tar2=convlist(base,k,h,w)
toc = time.time()
print(toc-tic)
tar2=np.array(tar2)
cv2.imwrite("d-blur-lib.jpg",tar)
cv2.imwrite("d-blur-my.jpg",tar2)