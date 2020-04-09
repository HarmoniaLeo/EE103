import cv2
import numpy as np

base = cv2.imread("base_iro.jpg")
holo=cv2.imread("holo.png")
base=base.astype(np.int32)
holo=holo.astype(np.int32)
tar=holo
tar=np.where(tar>255,255,tar)
tar=np.where(tar<0,0,tar)
cv2.imwrite("normal.jpg",tar)

tar1=np.where(base<holo,base,holo)
tar2=holo*base/255
tar3=np.where(holo==0,255,(holo+base-255)*255/holo)
tar4=holo+base-255
mask1=np.where(base[...,0]+base[...,1]+base[...,2]>holo[...,0]+holo[...,1]+holo[...,2],0,1)
mask2=np.where(base[...,0]+base[...,1]+base[...,2]>holo[...,0]+holo[...,1]+holo[...,2],1,0)
mask1=np.stack((mask1,mask1,mask1),axis=2)
mask2=np.stack((mask2,mask2,mask2),axis=2)
tar5=mask1*base+mask2*holo
tar=np.concatenate((tar1,tar2,tar3,tar4,tar5),axis=1)
tar=np.where(tar>255,255,tar)
tar=np.where(tar<0,0,tar)
cv2.imwrite("darken.jpg",tar)

tar1=np.where(base>holo,base,holo)
tar2=255-(255-holo)*(255-base)/255
tar3=np.where(base==255,0,base+holo*base/(255-base))
tar4=holo+base
tar5=mask2*base+mask1*holo
tar=np.concatenate((tar1,tar2,tar3,tar4,tar5),axis=1)
tar=np.where(tar>255,255,tar)
tar=np.where(tar<0,0,tar)
cv2.imwrite("brighten.jpg",tar)

tar=np.where(base<=128,base*holo/128,255-(255-holo)*(255-base)/128)
cv2.imwrite("stack.jpg",tar)

tar1=np.where(holo<=128,base+(2*holo-255)*(base-base*base/255)/255,\
    base+(2*holo-255)*(np.sqrt(base/255)-base)/255)
tar2=np.where(holo<=128,base*holo/128,255-(255-holo)*(255-base)/128)
tar3=np.where(holo<=128,255-(255-base)/(2*holo),base/(2*(255-holo))*255)
tar4=2*holo+base-255
tar=np.concatenate((tar1,tar2,tar3,tar4),axis=1)
tar=np.where(tar>255,255,tar)
tar=np.where(tar<0,0,tar)
cv2.imwrite("rays.jpg",tar)