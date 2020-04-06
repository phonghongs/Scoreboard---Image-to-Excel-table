import numpy as np
import cv2
import sys
import os
import random
import argparse
from supFunc import *
from openpyxl import Workbook

book = Workbook()
sheet = book.active

WIDTH_RESIZE = 2000

CL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image file")
    args = vars(ap.parse_args())
    image = cv2.imread(args['image'])
    # image = cv2.imread('Warped.jpg')

    y, x, chanel = image.shape
    image = cv2.resize(image,(WIDTH_RESIZE, int((y*WIDTH_RESIZE)/x)))
    copy_image = np.copy(image)

    firstMask_Image = FindTable(image)
    cont, hier = HoughLine(firstMask_Image)

    cont = sorted(cont, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)

    scoreBoardMatrix = Convert2Matrix(cont)

    # print(scoreBoardMatrix)

    
    
    for i, rowValue in enumerate(scoreBoardMatrix):
        for j, columValue in enumerate(rowValue):
            xS, yS, wS, hS, _, _ = columValue
            char = image[yS : yS + hS, xS : xS + wS]

            addr = str(CL[j]) + str(i + 1)
            if j==3:
                sheet[addr] = Name(char)
            elif j==9 or j==10:
                sheet[addr]=ModelPredict(char,i)
            elif j == 0:        #cột stt
                sheet[addr] = Sigle(char,True)
            elif j>=5:          #cột điểm
                sheet[addr] = Sigle(char)
            else:
                sheet[addr] = TesseractRecMulti(char)
            # cv2.imwrite("step/" + str(i) + "__" + str(j) + ".jpg", char)
    book.save("sample.xlsx")
    cv2.imwrite('step/afterSegment.jpg', copy_image)

cv2.waitKey(0)
cv2.destroyAllWindows()