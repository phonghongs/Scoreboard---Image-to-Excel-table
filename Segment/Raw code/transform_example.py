from transform import four_point_transform
import numpy as np 
import argparse
import cv2

scaleIndex = 0.2

refPt = []
RrefPt = []
def mouse_event(event, x, y, flags, param):
    global refPt

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        RrefPt.append((int(x / scaleIndex), int(y / scaleIndex)))

image = cv2.imread("Score-board.png")
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_event)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())
image = cv2.imread(args['image'])
# image = cv2.imread('Anh/test_5.jpg')

clone = image.copy()
Ch, Cw, Cchanel = clone.shape
clone = cv2.resize(clone, (int(Cw * scaleIndex), int(Ch * scaleIndex)))

while True:
    cv2.imshow("image", clone)

    for mouseP in refPt:
        cv2.circle(clone, (mouseP[0], mouseP[1]), 1, (0, 255, 0), 5)

    cv2.waitKey(1)
    if refPt.__len__() == 4:
        break

RrefPt = np.asarray(RrefPt)
warped = four_point_transform(image, RrefPt)

cv2.imshow("Original", image)

# for Rx, Ry in refPt:
#     cv2.circle(image, (int(Rx / scaleIndex), int(Ry / scaleIndex)), 1 , (0, 255, 0), 10)

cv2.imshow("Warped", warped)
cv2.imwrite("Warped.jpg", warped)

cv2.waitKey(0)