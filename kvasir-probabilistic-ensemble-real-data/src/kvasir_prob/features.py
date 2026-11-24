import cv2,numpy as np
def pixel_features(image,size=64):
    image=cv2.resize(image,(size,size));rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)/255.;hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)/255.
    yy,xx=np.mgrid[:size,:size];coords=np.stack([xx/(size-1),yy/(size-1)],-1)
    return np.concatenate([rgb,hsv,coords],-1).reshape(-1,8)

