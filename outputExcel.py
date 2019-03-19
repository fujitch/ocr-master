# -*- coding: utf-8 -*-

import xlwt
import ocr_functionset
import cv2
import math
import numpy as np

class excelGenerator:
    def __init__(self, image_path):
        self.functions = ocr_functionset.functions()
        self.image_path = image_path
        
    def generate():
        book = xlwt.Workbook()
        sheet1 = book.add_sheet('sheet1')
        
        sheet1.write(0, 0, 100)
        sheet1.write(0, 1, 200)
        sheet1.write(1, 0, 300)
        sheet1.write(1, 1, 400)
        
        book.save('test.xls')
        
    def test(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        thresh = 230
        max_pixel = 255
        ret, processedImg = cv2.threshold(gray,
                                          thresh,
                                          max_pixel,
                                          cv2.THRESH_BINARY)
        
        lines = cv2.HoughLinesP(edges, rho=5, theta=math.pi / 180, threshold=200, minLineLength=100, maxLineGap=10)
        for (x1, y1, x2, y2) in lines[0]:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imshow('score', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def selectGenerate():
        
        
if __name__=='__main__':
    excelGenerator = excelGenerator('choubotest2.png')
    excelGenerator.test()