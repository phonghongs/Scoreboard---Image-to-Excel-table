from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import ScoreboardUI

class SccoreBoardApp(QtWidgets.QMainWindow, ScoreboardUI.Ui_MainWindow):
    def __init__(self, parent=None):
        super(SccoreBoardApp, self).__init__(parent)
        self.setupUi(self)

def main():
    app = QApplication(sys.argv)
    form = SccoreBoardApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()