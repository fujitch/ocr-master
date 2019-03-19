# -*- coding: utf-8 -*-

import skimage.data
import selectivesearch
import cv2
import numpy as np
from sklearn.decomposition import PCA

# 合体を考慮する横幅中央値との比
side_rate = 0.7
length_rate = 0.7
# 角度を求める
def angle(x, y):

    dot_xy = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    cos = dot_xy / (norm_x*norm_y)
    rad = np.arccos(cos)
    theta = rad * 180 / np.pi

    return theta

# 文字が存在する領域の縦の長さを求める
def getLength(image, rowBound, colBound):
    count = 0
    for i in range(rowBound[0], rowBound[1]+1):
        if image[i, colBound[0]: colBound[1]].prod() == 0:
            count += 1
    return count
            

orgImg = cv2.imread('reiauto.jpg', cv2.IMREAD_COLOR)
img = orgImg
processedImg = orgImg

processedImg = np.ones((orgImg.shape[0], orgImg.shape[1]))*255
processedImg = processedImg.astype(np.uint8)
for i in range(processedImg.shape[0]):
    for k in range(processedImg.shape[1]):
        if orgImg[i, k, 0] < 10 and orgImg[i, k, 1] < 10 and orgImg[i, k, 0] < 10:
            processedImg[i, k] = 0

processedImg = cv2.cvtColor(processedImg, cv2.COLOR_BGR2GRAY)

thresh = 250
max_pixel = 255
ret, processedImg = cv2.threshold(processedImg,
                                  thresh,
                                  max_pixel,
                                  cv2.THRESH_BINARY)

"""
indexX, indexY = np.where(processedImg < 5)
index = np.array([indexX, indexY])
pca = PCA(n_components=2)
pca.fit(index.T)
theta1 = angle(pca.components_[0], [1, 0])
theta2 = angle(pca.components_[1], [1, 0])
size = tuple([processedImg.shape[1], processedImg.shape[0]])
center = tuple([int(size[0]/2), int(size[1]/2)])
if abs(90 - theta1) < abs(90 - theta2):
    if abs(180 - theta2) < abs(360 - theta2):
        angle = 180 - theta2
    else:
        angle = 360 - theta2
else:
    if abs(180 - theta1) < abs(360 - theta1):
        angle = 180 - theta1
    else:
        angle = 360 - theta1

scale = 1.0
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
processedImg = cv2.warpAffine(processedImg, rotation_matrix, size, flags=cv2.INTER_CUBIC)
"""

## 画像表示
cv2.imshow('image', processedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_lbl, regions = selectivesearch.selective_search(orgImg, scale=500, sigma=0.9, min_size=10)


density = np.zeros((img.shape[0]))

addFlags = np.ones((len(regions)))

for i in range(len(regions)):
    roi = regions[i]
    if roi['rect'][0] == 0 and roi['rect'][1] == 0 and roi['rect'][2] >= img.shape[1]-1 and roi['rect'][3] >= img.shape[0]-1:
        addFlags[i] = 0
        continue
    for k in range(len(regions)):
        if i == k:
            continue
        roi_com = regions[k]
        if roi['rect'][0] <= roi_com['rect'][0] and roi['rect'][1] <= roi_com['rect'][1] and roi['rect'][0] + roi['rect'][2] >= roi_com['rect'][0] + roi_com['rect'][2] and roi['rect'][1] + roi['rect'][3] >= roi_com['rect'][1] + roi_com['rect'][3]:
            addFlags[k] = 0
            
            
for i in range(len(regions)):
    roi = regions[i]
    """
    if addFlags[i] == 0:
        continue
    """
    orgPointX = roi['rect'][0]
    orgPointY = roi['rect'][1]
    otherPointX = roi['rect'][2]
    otherPointY = roi['rect'][3]
    
    img[orgPointY:orgPointY + otherPointY, orgPointX, 0] = np.uint8(255)
    img[orgPointY:orgPointY + otherPointY, orgPointX, 1] = np.uint8(0)
    img[orgPointY:orgPointY + otherPointY, orgPointX, 2] = np.uint8(0)
    density[orgPointY:orgPointY + otherPointY] += 1
    
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
cv2.waitKey(0)
cv2.destroyAllWindows()

rows = []
row = np.zeros((2))
row = row.astype(np.int32)
detectingFlg = False
for i in range(img.shape[0]):
    if i in[0, 1, img.shape[0]-1, img.shape[0]-2]:
        continue
    ## densityの閾値の決め方要検討
    if detectingFlg:
        if density[i] < 4:
            detectingFlg = False
            row[1] = i
            rows.append(row)
    else:
        if density[i] >= 4:
            detectingFlg = True
            row = np.zeros((2))
            row = row.astype(np.int32)
            row[0] = i


words = []
col = np.zeros((2))
col = col.astype(np.int)
detectingFlg = False
for row in rows:
    party = []
    colRange = []
    colList = []
    # 空白で区切って、それぞれ挿入
    for i in range(img.shape[1]):
        if i in[0, 1, img.shape[1]-1, img.shape[1]-2]:
            continue
        
        if detectingFlg:
            if processedImg[row[0]:row[1], i].prod() != 0:
                detectingFlg = False
                col[1] = i
                colRange.append(abs(col[1] - col[0]))   
                dummy = np.zeros((2))
                dummy = dummy.astype(np.int)
                dummy[0] = col[0]
                dummy[1] = col[1]
                colList.append(dummy)
        else:
            if processedImg[row[0]:row[1], i].prod() == 0:
                detectingFlg = True
                col[0] = i
                
    # 幅の中央値を求める
    colCenter = np.median(colRange)
    
    # 前後の文字の大きさから補正
    
    colListNew = []
    continueFlg = False
    newCol = np.zeros((2))
    newCol = newCol.astype(np.int)
    if len(colList) > 1:
        for i in range(len(colList)):
            # 1文字目
            if i == 0:
                col = colList[i]
                colNext = colList[i+1]
                # colNextと合体させる条件
                if colCenter*side_rate > abs(col[1] - col[0]) and colCenter*0.9 > abs(colNext[1] - colNext[0]):
                    # colPreを作成
                    colPre = np.zeros((2))
                    colPre = colPre.astype(np.int)
                    colPre[0] = col[0]
                    colPre[1] = colNext[1]
                    continueFlg = True
                    continue
                # 縦の幅が小さすぎし横も小さいときは文字の一部と考える。
                if getLength(processedImg, row, col) < abs(row[1] - row[0])*length_rate and colCenter*side_rate > abs(col[1] - col[0]):
                    # colPreを作成
                    colPre = np.zeros((2))
                    colPre = colPre.astype(np.int)
                    colPre[0] = col[0]
                    colPre[1] = colNext[1]
                    continueFlg = True
                    continue
                # 合体させない
                colPre = np.zeros((2))
                colPre = colPre.astype(np.int)
                colPre[0] = col[0]
                colPre[1] = col[1]
            # 最後の1文字
            elif i == len(colList) -1:
                # continueFlgが立っている場合、合体しているのでそのままcolPreを挿入して終わり
                if continueFlg:
                    # colPreを挿入
                    dummy = np.zeros((2))
                    dummy = dummy.astype(np.int)
                    dummy[0] = colPre[0]
                    dummy[1] = colPre[1]
                    colListNew.append(dummy)
                    continue
                col = colList[i]
                # 横幅が小さいと合体条件を考える
                if colCenter*side_rate > abs(col[1] - col[0]):
                    # colPreと合体させる条件
                    if abs(colPre[1] - colPre[0]) < colCenter*0.9:
                        colPre[1] = col[1]
                        # colPreを挿入
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colPre[0]
                        dummy[1] = colPre[1]
                        colListNew.append(dummy)
                        continue
                # 縦幅が小さすぎし横幅も小さいときは文字の一部と考える
                if getLength(processedImg, row, col) < abs(row[1] - row[0])*length_rate and colCenter*side_rate > abs(col[1] - col[0]):
                    colPre[1] = col[1]
                    # colPreを挿入
                    dummy = np.zeros((2))
                    dummy = dummy.astype(np.int)
                    dummy[0] = colPre[0]
                    dummy[1] = colPre[1]
                    colListNew.append(dummy)
                    continue
                # colPreを挿入
                dummy = np.zeros((2))
                dummy = dummy.astype(np.int)
                dummy[0] = colPre[0]
                dummy[1] = colPre[1]
                colListNew.append(dummy)
                # colも挿入
                dummy = np.zeros((2))
                dummy = dummy.astype(np.int)
                dummy[0] = col[0]
                dummy[1] = col[1]
                colListNew.append(dummy)
            # 二文字目～最後の一つ前の文字まで
            else:
                if continueFlg:
                    continueFlg = False
                    continue
                col = colList[i]
                colNext = colList[i+1]
                # 横幅が小さいと合体させる条件に入れる
                if colCenter*side_rate > abs(col[1] - col[0]):
                    # colPreと合体させる条件
                    if abs(colPre[1] - colPre[0]) < abs(colNext[1] - colNext[0]) and abs(colPre[1] - colPre[0]) < colCenter*0.9:
                        # colPreとcolを合体
                        colPre[1] = col[1]
                        continue
                    # colNextと合体させる条件
                    elif abs(colNext[1] - colNext[0]) < abs(colPre[1] - colPre[0]) and abs(colNext[1] - colNext[0]) < colCenter*0.9:
                        # colPreを挿入
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colPre[0]
                        dummy[1] = colPre[1]
                        colListNew.append(dummy)
                        # colPreにcolとcolNextを合体したものを挿入し、continueFlgを立てる
                        colPre[0] = col[0]
                        colPre[1] = colNext[1]
                        continueFlg = True
                        continue
                # 縦幅が小さすぎるし、横幅も小さめのものは何かの文字の一部とみなす。
                if getLength(processedImg, row, col) < abs(row[1] - row[0])*length_rate and colCenter*side_rate > abs(col[1] - col[0]):
                    # colPreと合体させる条件
                    if abs(colPre[1] - colPre[0]) < abs(colNext[1] - colNext[0]):
                        # colPreとcolを合体
                        colPre[1] = col[1]
                        continue                        
                    # colNextと合体させる条件
                    elif abs(colNext[1] - colNext[0]) < abs(colPre[1] - colPre[0]):
                        # colPreを挿入
                        dummy = np.zeros((2))
                        dummy = dummy.astype(np.int)
                        dummy[0] = colPre[0]
                        dummy[1] = colPre[1]
                        colListNew.append(dummy)
                        # colPreにcolとcolNextを合体したものを挿入し、continueFlgを立てる
                        colPre[0] = col[0]
                        colPre[1] = colNext[1]
                        continueFlg = True
                        continue
                # 問題なかったとき
                # colPreを挿入
                dummy = np.zeros((2))
                dummy = dummy.astype(np.int)
                dummy[0] = colPre[0]
                dummy[1] = colPre[1]
                colListNew.append(dummy)
                # colPreを更新
                colPre[0] = col[0]
                colPre[1] = col[1]
    colList = colListNew
    
    # 前後の文字の大きさから補正
    
    for col in colList:
        party.append(orgImg[row[0]:row[1], col[0]:col[1], :])

    # 従来のやつ     
    """
    for i in range(img.shape[1]):
        if detectingFlg:
            if processedImg[row[0]:row[1], i].prod() != 0:
                detectingFlg = False
                col[1] = i
                party.append(orgImg[row[0]:row[1], col[0]:col[1], :])
        else:
            if processedImg[row[0]:row[1], i].prod() == 0:
                detectingFlg = True
                col[0] = i
    """
        
    words.append(party)
                
for i in range(len(words)):
    party = words[i]
    for k in range(len(party)):
        word = party[k]
        if word.shape[1] == 0:
            continue
        ## 画像表示
        fname = 'each' + str(i) + '.jpg'
        cv2.imshow('image', word)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(fname, word)
    
    
## 画像表示
cv2.imshow('image', processedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('test.jpg', img)