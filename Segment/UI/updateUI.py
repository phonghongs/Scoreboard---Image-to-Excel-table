# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ScoreboardUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
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

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cImage.setText(_translate("MainWindow", "Choose Image"))
        self.start.setText(_translate("MainWindow", "Start"))
        self.save.setText(_translate("MainWindow", "Save"))
