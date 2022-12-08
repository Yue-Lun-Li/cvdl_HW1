from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2 , numpy as np
import matplotlib.pyplot as plt
from UI1_4 import Ui_MainWindow


class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()


    def setup_control(self):
        self.ui.btn_load_image_1.clicked.connect(self.load_image_1)
        self.ui.btn_load_image_2.clicked.connect(self.load_image_2)
        self.ui.btn_keypoints.clicked.connect(self.keypoints)
        self.ui.btn_matched_keypoints.clicked.connect(self.matched_keypoints)
    def load_image_1(self):
        self.img_1, _ = QFileDialog.getOpenFileName(
            self,
            filter='Image Files (*.png *.jpg *.jpeg *.bmp)')           # start path
        print(self.img_1)

    def load_image_2(self):
        self.img_2, _ = QFileDialog.getOpenFileName(
            self,
            filter='Image Files (*.png *.jpg *.jpeg *.bmp)')           # start path
        print(self.img_2)

    def keypoints(self):
        self.img_org = cv2.imread('{}'.format(self.img_1))
        sift = cv2.xfeatures2d.SIFT_create()
        self.kp_org, self.des_org = sift.detectAndCompute(self.img_org, None)
        self.sift_img_org = cv2.drawKeypoints(self.img_org, self.kp_org, self.img_org, color=(0,255,0))
        cv2.imwrite("figure1.png",self.sift_img_org)
        print("Keypoints is finished and saved")

    def matched_keypoints(self):

        img_scr = cv2.imread('{}'.format(self.img_2))



        sift = cv2.xfeatures2d.SIFT_create()
        kp_scr, des_scr = sift.detectAndCompute(img_scr, None)
        sift_img_scr = cv2.drawKeypoints(img_scr, kp_scr, img_scr, color=(0,255,0))
        sift_img_scr = np.vstack((sift_img_scr,np.zeros((self.img_org.shape[0]-img_scr.shape[0],img_scr.shape[1],3))))
        sift = np.hstack((self.sift_img_org,sift_img_scr))

        bf = cv2.BFMatcher()
        match = bf.knnMatch(self.des_org, des_scr, k = 2)
        ratio = 0.5
        good = []

        for m1, n1 in match:
            if m1.distance < ratio * n1.distance:
                good.append([m1])

        matched = cv2.drawMatchesKnn(self.img_org, self.kp_org, img_scr, kp_scr, good, None, flags = 2)
        cv2.imwrite("figure2.png", matched)
        print("Matched keypoints is finished and saved")