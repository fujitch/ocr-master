# -*- coding: utf-8 -*-

import selectivesearch
import cv2
import numpy as np

orgImg = cv2.imread('reiauto.jpg', cv2.IMREAD_COLOR)
img = orgImg

img_lbl, regions = selectivesearch.selective_search(orgImg, scale=500, sigma=0.9, min_size=10)

addFlags = np.ones((len(regions)))

for i in range(len(regions)):
    roi = regions[i]
    if roi['rect'][0] == 0 and roi['rect'][1] == 0 and roi['rect'][2] >= img.shape[1]-1 and roi['rect'][3] >= img.shape[0]-1:
        addFlags[i] = 0
        continue
    if addFlags[i] == 0:
        continue
    for k in range(len(regions)):
        if i == k:
            continue
        roi_com = regions[k]
        if roi['rect'][0] <= roi_com['rect'][0] and roi['rect'][1] <= roi_com['rect'][1] and roi['rect'][0] + roi['rect'][2] >= roi_com['rect'][0] + roi_com['rect'][2] and roi['rect'][1] + roi['rect'][3] >= roi_com['rect'][1] + roi_com['rect'][3]:
            addFlags[k] = 0
            if roi['rect'][0] == roi_com['rect'][0] and roi['rect'][1] == roi_com['rect'][1] and roi['rect'][0] + roi['rect'][2] == roi_com['rect'][0] + roi_com['rect'][2] and roi['rect'][1] + roi['rect'][3] == roi_com['rect'][1] + roi_com['rect'][3]:
                addFlags[i] = 1
            
            
for i in range(len(regions)):
    roi = regions[i]
    
    if addFlags[i] == 0:
        continue
    
    orgPointX = roi['rect'][0]
    orgPointY = roi['rect'][1]
    otherPointX = roi['rect'][2]
    otherPointY = roi['rect'][3]
    
    img[orgPointY:orgPointY + otherPointY, orgPointX, 0] = np.uint8(255)
    img[orgPointY:orgPointY + otherPointY, orgPointX, 1] = np.uint8(0)
    img[orgPointY:orgPointY + otherPointY, orgPointX, 2] = np.uint8(0)
    
    img[orgPointY:orgPointY + otherPointY, orgPointX + otherPointX, 0] = np.uint8(255)
    img[orgPointY:orgPointY + otherPointY, orgPointX + otherPointX, 1] = np.uint8(0)
    img[orgPointY:orgPointY + otherPointY, orgPointX + otherPointX, 2] = np.uint8(0)
    
    img[orgPointY, orgPointX:orgPointX + otherPointX, 0] = np.uint8(255)
    img[orgPointY, orgPointX:orgPointX + otherPointX, 1] = np.uint8(0)
    img[orgPointY, orgPointX:orgPointX + otherPointX, 2] = np.uint8(0)
    
    img[orgPointY + otherPointY, orgPointX:orgPointX + otherPointX, 0] = np.uint8(255)
    img[orgPointY + otherPointY, orgPointX:orgPointX + otherPointX, 1] = np.uint8(0)
    img[orgPointY + otherPointY, orgPointX:orgPointX + otherPointX, 2] = np.uint8(0)
    
    
## 画像表示
cv2.imshow('image', img)
keycode = cv2.waitKey(0)
if keycode == ord('s'): 
    cv2.imwrite("exaple.png", img)
cv2.destroyAllWindows()