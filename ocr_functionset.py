# -*- coding: utf-8 -*-

import selectivesearch
import cv2
import numpy as np
import random
from scipy import ndimage

class functions:
    def __init__(self):
        # 合体を考慮する横幅中央値との比
        self.side_rate = 0.7
        self.length_rate = 0.7
    # 角度を求める
    def angle(self, x, y):
        dot_xy = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        cos = dot_xy / (norm_x*norm_y)
        rad = np.arccos(cos)
        theta = rad * 180 / np.pi
        
        return theta
    
    # 文字が存在する領域の縦の長さを求める
    def getLength(self, image, rowBound, colBound):
        count = 0
        for i in range(rowBound[0], rowBound[1]+1):
            if image[i, colBound[0]: colBound[1]].prod() == 0:
                count += 1
        return count
    
    # 空白、大きさで区切って切り出した画像リストを返す
    def convert(self, image_path):
        # 画像読み込み
        orgImg = cv2.imread(image_path, cv2.IMREAD_COLOR)
        originalImg = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = orgImg
        processedImg = orgImg
        # グレースケールに変換
        processedImg = cv2.cvtColor(processedImg, cv2.COLOR_BGR2GRAY)

        # 二値化処理
        thresh = 250
        max_pixel = 255
        ret, processedImg = cv2.threshold(processedImg,
                                          thresh,
                                          max_pixel,
                                          cv2.THRESH_BINARY)
        # selectiveSearch適用
        img_lbl, regions = selectivesearch.selective_search(orgImg, scale=500, sigma=0.9, min_size=10)
        
        # 注目領域の密度を求める
        density = np.zeros((img.shape[0]))
        for i in range(len(regions)):
            roi = regions[i]
            
            if len(roi['labels']) != 1:
                continue
            
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
            
        # 横に区切る
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
               
        # 縦にも区切って1枚1枚の画像リストを作る
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
                        if colCenter*self.side_rate > abs(col[1] - col[0]) and colCenter*0.9 > abs(colNext[1] - colNext[0]):
                            # colPreを作成
                            colPre = np.zeros((2))
                            colPre = colPre.astype(np.int)
                            colPre[0] = col[0]
                            colPre[1] = colNext[1]
                            continueFlg = True
                            continue
                        # 縦の幅が小さすぎし横も小さいときは文字の一部と考える。
                        if self.getLength(processedImg, row, col) < abs(row[1] - row[0])*self.length_rate and colCenter*self.side_rate > abs(col[1] - col[0]):
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
                        if colCenter*self.side_rate > abs(col[1] - col[0]):
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
                        if self.getLength(processedImg, row, col) < abs(row[1] - row[0])*self.length_rate and colCenter*self.side_rate > abs(col[1] - col[0]):
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
                        if colCenter*self.side_rate > abs(col[1] - col[0]):
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
                        if self.getLength(processedImg, row, col) < abs(row[1] - row[0])*self.length_rate and colCenter*self.side_rate > abs(col[1] - col[0]):
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
                party.append(originalImg[row[0]:row[1], col[0]:col[1], :])
                
            words.append(party)
        return words
    # 学習画像の加工（回転、リサイズ、グレースケール化、二値化の順）
    def process_img(self, img, size, center, scale, thresh, original_image_size):
        # 回転処理
        angle = 10 * (random.random() - 0.5)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        processedImg = cv2.warpAffine(img, rotation_matrix, size, flags=cv2.INTER_CUBIC)
        # 回転の隙間埋め
        processedImg[0:10, :, :] = 255
        processedImg[original_image_size - 10:original_image_size, :, :] = 255
        processedImg[:, 0:10, :] = 255
        processedImg[:, original_image_size - 10:original_image_size, :] = 255
        # ランダムで拡大縮小
        new_size = random.randint(original_image_size*7/11, original_image_size*14/11)
        processedImg = cv2.resize(processedImg, (new_size, new_size))
        # ランダムで位置を移動
        if new_size >= original_image_size:
            center_px = new_size / 2
            new_image = processedImg[center_px - original_image_size/2: center_px + original_image_size/2, center_px - original_image_size/2: center_px + original_image_size/2, :]
        else:
            center_px = random.randint(0, original_image_size - new_size)
            center_py = random.randint(0, original_image_size - new_size)
            new_image = np.ones((original_image_size, original_image_size, 3)) * 255
            new_image = new_image.astype(np.uint8)
            new_image[center_px: center_px + new_size, center_py: center_py + new_size, :] = processedImg
            
        
        # グレースケール化と二値化、白黒化、リサイズ
        img_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        # _, img_gray = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
        img_gray = 255 - img_gray
        img_gray = cv2.resize(img_gray, None, fx = 0.5, fy = 0.5)
        img_gray = img_gray.astype(np.float32) / 255
        
        return img_gray
    
    # 学習画像の加工（回転、リサイズ、グレースケール化、二値化の順）
    def process_img_scy(self, img, thresh, original_image_size):
        # 回転処理
        angle = 10 * (random.random() - 0.5)
        processedImg = ndimage.rotate(img, angle, reshape=False)
        # 回転の隙間埋め
        processedImg[0:10, :, :] = 255
        processedImg[original_image_size - 10:original_image_size, :, :] = 255
        processedImg[:, 0:10, :] = 255
        processedImg[:, original_image_size - 10:original_image_size, :] = 255
        # ランダムで拡大縮小
        new_size = random.randint(original_image_size*7/11, original_image_size*14/11)
        processedImg = cv2.resize(processedImg, (new_size, new_size))
        # ランダムで位置を移動
        if new_size >= original_image_size:
            center_px = new_size / 2
            new_image = processedImg[center_px - original_image_size/2: center_px + original_image_size/2, center_px - original_image_size/2: center_px + original_image_size/2, :]
        else:
            center_px = random.randint(0, original_image_size - new_size)
            center_py = random.randint(0, original_image_size - new_size)
            new_image = np.ones((original_image_size, original_image_size, 3)) * 255
            new_image = new_image.astype(np.uint8)
            new_image[center_px: center_px + new_size, center_py: center_py + new_size, :] = processedImg
            
        
        # グレースケール化と二値化、白黒化、リサイズ
        img_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        # _, img_gray = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
        img_gray = 255 - img_gray
        img_gray = cv2.resize(img_gray, None, fx = 0.5, fy = 0.5)
        img_gray = img_gray.astype(np.float32) / 255
        
        return img_gray
    
    # selective_searchで行ごとに区切った画像リストを返す
    def convert_row(self, image_path):
        # 画像読み込み
        orgImg = cv2.imread(image_path, cv2.IMREAD_COLOR)
        originalImg = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = orgImg
        processedImg = orgImg
        # グレースケールに変換
        processedImg = cv2.cvtColor(processedImg, cv2.COLOR_BGR2GRAY)

        # 二値化処理
        thresh = 230
        max_pixel = 255
        ret, processedImg = cv2.threshold(processedImg,
                                          thresh,
                                          max_pixel,
                                          cv2.THRESH_BINARY)
        # selectiveSearch適用
        img_lbl, regions = selectivesearch.selective_search(orgImg, scale=500, sigma=0.9, min_size=10)
        
        # 注目領域の密度を求める
        density = np.zeros((img.shape[0]))
        for i in range(len(regions)):
            roi = regions[i]
            
            if len(roi['labels']) != 1:
                continue
            
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
            
        # 横に区切る
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
        images = []
        processed = []
        for row in rows:
            images.append(originalImg[row[0]:row[1], :, :])
            processed.append(processedImg[row[0]:row[1], :])
        
        return images, processed
    
    def simply_cut_col(self, processedImg):
        col = np.zeros((2))
        col = col.astype(np.int)
        detectingFlg = False
        colList = []
        # 空白で区切って、それぞれ挿入
        for i in range(processedImg.shape[1]):
            if i in[0, 1, processedImg.shape[1]-1, processedImg.shape[1]-2]:
                continue
            
            if detectingFlg:
                if processedImg[:, i].prod() != 0:
                    detectingFlg = False
                    col[1] = i
                    dummy = np.zeros((2))
                    dummy = dummy.astype(np.int)
                    dummy[0] = col[0]
                    dummy[1] = col[1]
                    colList.append(dummy)
            else:
                if processedImg[:, i].prod() == 0:
                    detectingFlg = True
                    col[0] = i
        
        return colList