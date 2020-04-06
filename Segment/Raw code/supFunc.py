from PIL import Image
import cv2
import numpy as np
import os
import pytesseract 
from keras.models import load_model
WIDTH_RESIZE = 2000
DEBUG = True
LIMIT_PIXEL = 4200
MAX_WIDTH = 1300
MAX_HIGHT = 90

MIN_HIGHT = 20
MIN_WIDTH = 20
model=load_model('model-final.h5')

def count_contour(image_f,indexx):
    h,w,c=image_f.shape   

    high=float(h/3.0)
    
    image_ff=image_f.copy
    image_ff = cv2.cvtColor(image_f, cv2.COLOR_BGR2GRAY)
    
    ret, image_ff = cv2.threshold(image_ff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    image_ff = cv2.morphologyEx(image_ff,cv2.MORPH_DILATE,kerel3)
    check=0
    for i in range(0,3):       
        image_c=image_ff[int(i*high):int((i+1)*high),0:w] #lấy 1/3 tấm hình
        c, hh = cv2.findContours(image_c,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #tìm contour
        count=0
        #đếm contour
        for j in c:
            count=count+1
        cv2.imwrite("step/"+str(indexx)+"__"+str(i)+"__"+str(count)+"__"+".jpg", image_c)
        if count is 1:  #nếu chỉ có 1 contour thì check tăng 1
            check=check+1
    if check is 1:      #nếu trong 3 phần chỉ có 1 phần 1 contour thì đó là ảnh ghép
        return True
    return False

def HoughLine(imgH):
    gray = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY) # color -> gray
    ret, threH = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    threH = cv2.bitwise_not(threH)

    mask_H = np.zeros(imgH.shape, np.uint8)
    lines = cv2.HoughLines(threH, rho=0.001, theta=np.pi/90, threshold=150)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
        cv2.line(mask_H, (x1, y1), (x2, y2), (255, 255, 255), 1) 
    print(mask_H.shape)

    mask_H = cv2.cvtColor(mask_H, cv2.COLOR_BGR2GRAY)
    ret_mask, thre_mask = cv2.threshold(mask_H, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    thre_mask = cv2.bitwise_not(thre_mask)
    kerel5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    thre_mask = cv2.morphologyEx(thre_mask,cv2.MORPH_DILATE,kerel5)

    #Thu nhỏ line, lấy Houghline
    kerel9 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    thre_mask = cv2.morphologyEx(thre_mask,cv2.MORPH_ERODE,kerel9)

    y_f , x_f = thre_mask.shape
    img = cv2.rectangle(thre_mask, (0,0), (x_f, y_f), 1, 3)

    cv2.imwrite("step/threshMask.jpg", thre_mask)
    
    cont, hier = cv2.findContours(thre_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return cont, hier

def FindTable(img):
    y_rs, x_rs, chanel_rs = img.shape
    img = cv2.rectangle(img, (0,0), (x_rs, y_rs), (0, 0, 0), 3)
    copy_image = np.copy(img)

    roi_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if DEBUG:
        cv2.imwrite("step/gray.jpg", roi_gray)

    ret, thre = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imwrite("step/thre.jpg", thre)

    '''Thuật toán dilate
    Xem ví dụ tại: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html'''

    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    thre_mor = cv2.morphologyEx(thre,cv2.MORPH_DILATE,kerel3)

    # Tìm tất cả các contours trên ảnh
    mask = np.zeros(img.shape, np.uint8)
    cont, hier = cv2.findContours(thre_mor,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(mask_F, cont, -1, 255, -1)

    if DEBUG:
        cv2.imwrite("step/mask.jpg", mask)

    for ind,cnt in enumerate(cont) :
        xR,yR,wR,hR = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if (area > LIMIT_PIXEL) & (wR < MAX_WIDTH) & (hR < MAX_HIGHT) & (wR > MIN_WIDTH) & (hR > MIN_HIGHT):
            mask = cv2.rectangle(mask, (xR, yR), (xR+wR, yR+hR), (0, 0, 255), 1)

    return mask

def Convert2Matrix(contour):
    charNum = 0
    rowA = []
    scoreBoardMatrix = []
    POSITION_SORT = []
    
    for indexS, cntS in enumerate(contour) :
        xS, yS, wS, hS = cv2.boundingRect(cntS)
        # copy_image = cv2.rectangle(copy_image, (xS, yS), (xS + wS, yS + hS), (0, 0, 255), 1)
        area = cv2.contourArea(cntS)
        if (area > LIMIT_PIXEL) & (wS < MAX_WIDTH) & (hS < MAX_HIGHT) & (wS > MIN_WIDTH) & (hS > MIN_HIGHT) :
            Center = [int((2*xS + wS)/2), int((2*yS + hS)/2)]
            if (Center[1] > 150):
                POSITION_SORT.append([xS, yS, wS, hS, Center[0], Center[1]])
                # copy_image = cv2.circle(copy_image, (Center[0], Center[1]), 2, (0 , 0 , 255), 3)   
                if Center[1] not in rowA:
                    rowA.append(Center[1])

    POSITION_SORT = sorted(POSITION_SORT, key= lambda area_Index: area_Index[5] , reverse = False)
    rowA = sorted(rowA, reverse = False)

    rowIndex = 0
    rowSortCache = []

    for indMa, sortMa in enumerate(POSITION_SORT):
        x_Ma, y_Ma, w_Ma, h_Ma, colum_Ma, row_Ma = sortMa
        
        if (row_Ma == rowA[rowIndex]):
            rowSortCache.append(sortMa)
        else:
            rowSortCache = sorted(rowSortCache, key= lambda area_Index: area_Index[4] , reverse = False)
            scoreBoardMatrix.append(rowSortCache)
            rowSortCache = []
            rowSortCache.append(sortMa)
            rowIndex += 1
        
    rowSortCache = sorted(rowSortCache, key= lambda area_Index: area_Index[4] , reverse = False)
    scoreBoardMatrix.append(rowSortCache)
    rowSortCache = []

    return scoreBoardMatrix
def Process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),1)
    ret, thre = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thre

def ResizeImg(img,w1,h1):         
    if w1<h1: 
        char_digit=cv2.resize(img,(int(float(28/h1)*w1),28))       
        mask = np.zeros((28,28-int(float(28/h1)*w1)), np.uint8)
        thresh = cv2.hconcat([char_digit, mask])
        trans_x = 14-int(int(float(28/h1)*w1)/2)
        trans_y = 0
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = thresh.shape[:2]
        thresh1 = cv2.warpAffine(thresh, trans_m, (width, height))
        return thresh1
    else:
        height, width = img.shape[:2]
        char_digit=cv2.resize(img,(28,int(float(28/w1)*h1)))       
        mask = np.zeros((28-int(float(28/w1)*h1),28), np.uint8)
        thresh = cv2.vconcat([char_digit, mask])
        trans_x = 0
        trans_y = 14-int(int(float(28/w1)*h1)/2)
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = thresh.shape[:2]
        thresh1 = cv2.warpAffine(thresh, trans_m, (width, height))
        return thresh1

def TakeX(elem):
    return elem[0]

def ModelPredict(char,i):
    roi = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi,(3,3),1)
    kernel = np.ones((1,1),np.uint8)
    roi = cv2.erode(roi,kernel,iterations = 1)
    _,roi = cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    yT, xT = roi.shape[0:2]
    roi = cv2.rectangle(roi, (0, 0), (xT, yT), (0, 0, 0), 7)
    cv2.imwrite("step/"+str(i)+".jpg", roi)
    cont_char, hier = cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    count_number=0
    coordinates=[]
    for index,cnt_b in enumerate(cont_char):        
        if cv2.contourArea(cnt_b)<100:              #Bỏ hết trường hợp nhiễu
            continue
        x1,y1,w1,h1 = cv2.boundingRect(cnt_b)
        clone=char.copy()       
        char_digit=clone[y1:y1+h1, x1:x1+w1]        
        
        if w1>=h1:      #kiểm tra xem đây có phải là hai số viết liền nhau
            if count_contour(char_digit,i) is True:         #trường hợp hai số viết liền nhau
                count_number=count_number+2
                coordinates.append([x1,y1,int(w1/2),h1])
                coordinates.append([x1+int(w1/2),y1,int(w1/2),h1])
            else:                                           #trường hợp số đơn
                count_number=count_number+1
                coordinates.append([x1,y1,w1,h1])            
        else:                                               #trường hợp số đơn
            count_number=count_number+1 
            coordinates.append([x1,y1,w1,h1])
    coordinates.sort(key=TakeX)
    pred=""
    if len(coordinates) is 3:
        x1,y1,w1,h1=coordinates[0]
        img1=char[y1:y1+h1,x1:x1+w1]
        img1=Process(img1)
        img1=ResizeImg(img1,w1,h1)
        y_predict1 = model.predict(img1.reshape(1,28,28,1))
        x2,y2,w2,h2=coordinates[1]
        img2=char[y2:y2+h2,x2:x2+w2]
        img2=Process(img2)
        img2=ResizeImg(img2,w2,h2)
        y_predict2 = model.predict(img2.reshape(1,28,28,1))
        if y_predict2[0][2]>y_predict2[0][7]:
            return str(np.argmax(y_predict1))+".25"
        else:
            return str(np.argmax(y_predict1))+".75"
    elif len(coordinates) is 2:
        x,y,w,h=coordinates[1]
        img=char[y:y+h,x:x+w]
        img=Process(img)
        img=ResizeImg(img,w,h)
        y_predict = model.predict(img.reshape(1,28,28,1))
        
        if y_predict[0][0]>y_predict[0][5] :

            cv2.imwrite("step/"+str(i)+"__"+"__"+str(np.argmax(y_predict))+".jpg", img)
            
            return "10"
        else:
            x,y,w,h=coordinates[0]
            img=char[y:y+h,x:x+w]
            img=Process(img)
            img=ResizeImg(img,w,h)
            y_predict = model.predict(img.reshape(1,28,28,1))
            cv2.imwrite("step/"+str(i)+"__"+"__"+str(np.argmax(y_predict))+".jpg", img)
            
            return str(np.argmax(y_predict))+".5"
        # cv2.imwrite("step/"+str(i)+"__"+str(cv2.contourArea(cnt_b))+"__"+str(np.argmax(y_predict))+".jpg", img)
    for arr in coordinates:
        x,y,w,h=arr
        img=char[y:y+h,x:x+w]
        img=Process(img)
        img=ResizeImg(img,w,h)
        y_predict = model.predict(img.reshape(1,28,28,1))
        pred=pred+str(np.argmax(y_predict))
        cv2.imwrite("step/"+str(i)+"__"+str(cv2.contourArea(cnt_b))+"__"+str(np.argmax(y_predict))+".jpg", img)
              
    return pred

def TesseractRecSingler(image,Str=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    copy_img = np.copy(image)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    yT, xT = thresh.shape[0:2]

    thresh = cv2.rectangle(thresh, (0, 0), (xT, yT), (255, 255, 255), 5)

    thresh = cv2.bitwise_not(thresh)
    h,w=thresh.shape

    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(13, 13))
    thre_mor = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kerel3)

    cont, hier = cv2.findContours(thre_mor,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h,w*2), np.uint8)
    # cv2.imshow("mask", mask)
    xR,yR,wR,hR=0,0,0,0
    for ind,cnt in enumerate(cont) :
        xR,yR,wR,hR = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        pre=thresh[yR-2:yR+hR+2,xR-2:xR+wR+2]
        if area > 500: 
            copy_img = cv2.rectangle(copy_img, (xR, yR), (xR+wR, yR+hR), (0, 255, 0), 3)
            break
        
    trans_x = w-wR-xR
    trans_y = 0
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    thresh1 = cv2.warpAffine(thresh, trans_m, (width, height))
    # cv2.imshow("mask1", thresh1)
    trans_x = -xR
    trans_y = 0
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    thresh2 = cv2.warpAffine(thresh, trans_m, (width, height))
    # cv2.imshow("mask2", thresh2)
    thresh = cv2.hconcat([thresh1, thresh2])

    # cv2.imshow("cop",copy_img)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kerel3)
    thresh = cv2.bitwise_not(thresh)	
    if Str==True:
        text = pytesseract.image_to_string(Image.fromarray(thresh), lang='vie')
    else:
        # text = pytesseract.image_to_string(Image.fromarray(thresh), lang='eng')
        config = ('--psm 10')
        text = pytesseract.image_to_string(Image,config='--psm 10')

    return text

def Sigle(image,Stt=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    copy_img = np.copy(image)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    yT, xT = thresh.shape[0:2]

    thresh = cv2.rectangle(thresh, (0, 0), (xT, yT), (255, 255, 255), 5)
    if Stt is True: #cột stt
        text = pytesseract.image_to_string(thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        if text is "":
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
            thresh = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kerel3)
            return pytesseract.image_to_string(thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        return text
    else:   #cột điểm
        text = pytesseract.image_to_string(thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.')
        if text is "":
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
            thresh = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kerel3)
            return pytesseract.image_to_string(thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.')
        return text

def Name(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    copy_img = np.copy(image)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    yT, xT = thresh.shape[0:2]
    # -c tessedit_char_whitelist=qwertyuiopasdfghjklzxcvbnmâăôơưêạãảáàấầẩẫậắằẳẵặúùủũụứữừựửắằẳẵặóòỏõọốồổỗộớờỡởợéèẻẽẹếềễểệQWERTYUIOPASDFGHJKLZXCVBNMÁÀẢÃẠÂẤẦẪẨẬĂẮẰẲẴẶƯỨỪỬỮỰÚÙỦŨỤÓÒỎÕỌÔỐỒỖỔỘƠỚỜỠỞỢÉÈẺẼẸÊẾỀỂỄỆâăưêôơ'
    thresh = cv2.rectangle(thresh, (0, 0), (xT, yT), (255, 255, 255), 5)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
    text = pytesseract.image_to_string(thresh, config='-l vie --psm 7 --oem 3 ')
    listText=list(text)
    for char in listText:
        if (ord(char)>=0 and ord(char)<65) or (ord(char)>=91 and ord(char)<97) or (ord(char)>=123 and ord(char)<256):
            return pytesseract.image_to_string(cv2.morphologyEx(thresh,cv2.MORPH_DILATE,kerel3), config='-l vie --psm 7 --oem 3 ')

    return text

def TesseractRecMulti(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    copy_img = np.copy(image)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    yT, xT = thresh.shape[0:2]

    thresh = cv2.rectangle(thresh, (0, 0), (xT, yT), (255, 255, 255), 5)

    # Load ảnh và apply nhận dạng bằng Tesseract OCR
    text = pytesseract.image_to_string(Image.fromarray(thresh),lang='vie')
    # os.remove(filename)

    return text