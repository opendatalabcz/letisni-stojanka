# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from cv2 import VideoWriter, VideoWriter_fourcc
from PIL import Image
import os
from os import listdir
import sys
import numpy as np
import pandas as pd
import time


CONFIG_PATH = "model/data.cfg"
WEIGHTS_PATH = "model/weights.weights"
LABELS_PATH = "model/data.names"

class Inferencer:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.LABELS = open(LABELS_PATH).read().strip().split("\n")

    def inference(self, image_path):
        image = np.asarray(Image.open(image_path))

        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        # Initializing for getting box coordinates, confidences, classid 
        boxes = []
        confidences = []
        classIDs = []
        threshold = 0.15

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")           
                    #x = int(centerX - (width / 2))
                    #y = int(centerY - (height / 2))    
                    boxes.append([centerX, centerY, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        print(classIDs)           
        for i in classIDs:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            for label in self.LABELS:
                if self.LABELS[classIDs[i]] == label:
                    color = (0, 0, 0)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                    text = "{}".format(self.LABELS[classIDs[i]])
                    cv2.putText(image, text, (x + w, y + h),                  
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)
                
        return image


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(20, 70, 500, 300))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap("test_imgs/frame_000090.jpg"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(170, 380, 131, 18))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.inference_image)

        self.Inferencer = Inferencer()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Detect"))

    def change_image(self):
        self.photo.setPixmap(QtGui.QPixmap("test_imgs/frame_000090.jpg"))

    def inference_image(self):
        image = self.Inferencer.inference("test_imgs/frame_000090.jpg")
        
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        
        qImg = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        
        self.photo.setPixmap(pixmap_image)
        print(self.photo.pixmap)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
