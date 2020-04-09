import cv2
import numpy as np
import sys 

R=[3,5,15]

def conv(filter,inputs):
    H, W = inputs.shape
    result = np.zeros(inputs.shape)
    H, W = inputs.shape
    for r in range(0, H - filter.shape[0] + 1):
        for c in range(0, W - filter.shape[1] + 1):
            cur_input = inputs[r:r + filter.shape[0],c:c + filter.shape[1]]
            cur_output = cur_input * filter
            conv_sum = np.sum(cur_output)
            result[r, c] = conv_sum
    return result

def med(inputs,R):
    H, W = inputs.shape
    result = np.zeros(inputs.shape)
    H, W = inputs.shape
    for r in range(0, H - R + 1):
        for c in range(0, W - R + 1):
            cur = inputs[r:r + R,c:c + R]
            val = np.median(cur)
            result[r, c] = val
    return result

#对图像添加10%椒盐噪声
base = cv2.imread("base.jpg",cv2.IMREAD_GRAYSCALE)
base=base.astype(np.int32)
noise=np.random.randint(0,100,base.shape)
base=np.where(noise<5,0,base)
base=np.where(noise>=95,255,base)
cv2.imwrite("noised.jpg",base)

tars=[]
for r in R:
    kernel=np.zeros((r,r))
    kernel+=1/r/r
    tar=conv(kernel,base)
    tar.resize((1200,1200))
    tars.append(tar)
cv2.imwrite("result-avg.jpg",np.concatenate(tars,1))
print("ok")
tars=[]
for r in R:
    tar=med(base,r)
    tar.resize((1200,1200))
    tars.append(tar)
cv2.imwrite("result-med.jpg",np.concatenate(tars,1))