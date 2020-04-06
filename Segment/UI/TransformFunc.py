from transformsupF import four_point_transform
import numpy as np 
import cv2

scaleIndex = 0.2

refPt = []
RrefPt = []
def mouse_event(event, x, y, flags, param):
    global refPt

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        RrefPt.append((int(x / scaleIndex), int(y / scaleIndex)))

def FourPoinTF(OriginalImage):  #Four point transform Function
    global refPt
    global RrefPt
    refPt = []
    RrefPt = []
    cv2.namedWindow("Select 4 corners of The Scoreboard")
    cv2.setMouseCallback("Select 4 corners of The Scoreboard", mouse_event)

    clone = np.copy(OriginalImage)
    Ch, Cw, Cchanel = clone.shape
    clone = cv2.resize(clone, (int(Cw * scaleIndex), int(Ch * scaleIndex)))

    while True:
        cv2.imshow("Select 4 corners of The Scoreboard", clone)

        for mouseP in refPt:
            cv2.circle(clone, (mouseP[0], mouseP[1]), 1, (0, 255, 0), 5)

        cv2.waitKey(1)
        if refPt.__len__() == 4:
            break

    RrefPt = np.asarray(RrefPt)
    warped = four_point_transform(OriginalImage, RrefPt)
    cv2.destroyAllWindows()

    return warped

def Choose_HWriting_Colum(image, scoreBoardMatrix):
    global refPt
    global RrefPt
    refPt = []
    RrefPt = []

    cv2.namedWindow("Handwriting columns (press [q] to next)")
    cv2.setMouseCallback("Handwriting columns (press [q] to next)", mouse_event)
    clone = np.copy(image)

    Ch, Cw, Cchanel = clone.shape
    clone = cv2.resize(clone, (int(Cw * scaleIndex), int(Ch * scaleIndex)))

    positionArray = []

    while True:
        cv2.imshow("Handwriting columns (press [q] to next)", clone)

        for mouseP in refPt:
            cv2.circle(clone, (mouseP[0], mouseP[1]), 1, (0, 255, 0), 5)

        if (cv2.waitKey(1) == ord('q')):
            break
    
    sBAr = []
    Ac = scoreBoardMatrix[0] 
    for Bc in Ac:
        sBAr.append([Bc[0], Bc[0] + Bc[2]])

    handWrColum = []
    for colums in RrefPt:
        for index, position in enumerate(sBAr):
            if colums[0] in range(position[0], position[1]):
                handWrColum.append(index)
                break
    
    cv2.destroyAllWindows()
    return handWrColum
    