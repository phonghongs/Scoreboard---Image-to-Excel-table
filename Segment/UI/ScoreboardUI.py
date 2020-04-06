# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ScoreboardUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QFileDialog, QLineEdit
from PyQt5.QtGui import QIcon
import sys
import Filedialog as fd
import cv2
import numpy as np 
from TransformFunc import *
from supFunc import *
from openpyxl import Workbook

class Ui_MainWindow(object):
    
    processCount = 0
    OriginalScoreB = []
    ResizeScoreB = []
    book = Workbook()
    sheet = book.active
    CL = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(412, 208)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cImage = QtWidgets.QPushButton(self.centralwidget)
        self.cImage.setGeometry(QtCore.QRect(20, 10, 91, 31))
        self.cImage.setObjectName("cImage")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(160, 10, 91, 31))
        self.start.setObjectName("start")
        self.save = QtWidgets.QPushButton(self.centralwidget)
        self.save.setGeometry(QtCore.QRect(300, 10, 91, 31))
        self.save.setObjectName("save")
        self.Terminal = QtWidgets.QTextBrowser(self.centralwidget)
        self.Terminal.setGeometry(QtCore.QRect(20, 70, 371, 81))
        self.Terminal.setObjectName("Terminal")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 412, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Terminal.append("--> Please [Choose Image] <--")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cImage.setText(_translate("MainWindow", "Choose Image"))
        self.start.setText(_translate("MainWindow", "Start"))
        self.save.setText(_translate("MainWindow", "Save"))

    def openDialog(self):
        self.Terminal.append("_______   New Image  _______")
        self.Terminal.append("--> Select your image path: ")

        ex = fd.AppOpen()

        if (ex.pathImage != ""):
            self.processCount = 1

            self.OriginalScoreB = cv2.imread(ex.pathImage)
            self.ResizeScoreB = FourPoinTF(self.OriginalScoreB)
            self.Terminal.append("Your path: " + ex.pathImage)
            

    def StartProcess(self):
        if (self.processCount == 1):

            self.Terminal.append("_______   Start Main Process ________")

            image = self.ResizeScoreB
            y, x, chanel = image.shape
            image = cv2.resize(image,(WIDTH_RESIZE, int((y*WIDTH_RESIZE)/x)))
            self.ResizeScoreB = image
            copy_image = np.copy(image)

            firstMask_Image = FindTable(image)

            self.Terminal.append("--> Find Table (done 1/5)")
            self.Terminal.append("--> Waiting, it might take some secs")

            cont, hier = HoughLine(firstMask_Image)
            cont = sorted(cont, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)

            self.Terminal.append("--> (done 2/5)")

            scoreBoardMatrix = Convert2Matrix(cont)

            self.Terminal.append("--> Convert Word position to Matrix (done 3/5)")

            handWrColumn = Choose_HWriting_Colum(image, scoreBoardMatrix)

            self.Terminal.append("--> What column that have Numbers Handwriting (done 4/5)")
            self.Terminal.append("--> Waiting, it might take some secs")


            for i, rowValue in enumerate(scoreBoardMatrix):
                for j, columValue in enumerate(rowValue):
                    xS, yS, wS, hS, _, _ = columValue
                    char = image[yS : yS + hS, xS : xS + wS]
                    addr = str(self.CL[j]) + str(i + 1)

                    if j==3:
                        self.sheet[addr] = Name(char)
                    elif j == 0:        #cột stt
                        self.sheet[addr] = Sigle(char,True)
                    elif j <= 4:
                        self.sheet[addr] = TesseractRecMulti(char)
                    elif j in handWrColumn:
                        self.sheet[addr]=ModelPredict(char,i)
                    elif j >= 5:          #cột điểm
                        self.sheet[addr] = Sigle(char)

            self.Terminal.append("--> Predict done (done 5/5)")
            
            self.processCount = 2

    def SaveFunc(self):
        if self.processCount == 2:
            self.Terminal.append("______  Save  ______")
            ex = fd.AppSave()
            fileName = ex.pathSave
            if fileName == "":
                self.Terminal.append("===> Please write your save path")
            else:
                self.book.save(fileName)
                self.Terminal.append("_____  End, you can choose new Image  ______")