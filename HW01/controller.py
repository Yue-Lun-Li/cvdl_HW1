from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
import cv2 , numpy as np
import matplotlib.pyplot as plt
from UI import Ui_MainWindow



class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.number = 1
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        self.corners_vertical = 8
        self.corners_horizontal = 11
        self. pattern_size = (self.corners_horizontal, self.corners_vertical)
        self.world_point = np.zeros((self.corners_vertical * self.corners_horizontal, 3), np.float32)
        self.world_point[:, :2] = np.mgrid[:self.corners_horizontal, :self.corners_vertical].T.reshape(-1, 2)
        self.world_points = []
        self.img_points = []
        self.axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1,3)

    def setup_control(self):
        self.ui.btn_load_folder.clicked.connect(self.load_folder)
        self.ui.btn_load_image_l.clicked.connect(self.load_image_L)
        self.ui.btn_load_image_r.clicked.connect(self.load_image_R)
        self.ui.btn_find_corners.clicked.connect(self.find_corners)
        self.ui.btn_find_intrinsic.clicked.connect(self.find_intrinsic)
        self.ui.bmp_number.currentIndexChanged.connect(self.numbers)
        self.ui.btn_find_extrinsic.clicked.connect(self.find_extrinsic)
        self.ui.btn_find_distortion.clicked.connect(self.find_distortion)
        self.ui.btn_show_result.clicked.connect(self.show_result)

        self.ui.btn_words_on_board.clicked.connect(self.words_on_board)
        self.ui.btn_words_vertically.clicked.connect(self.words_vertically)

        self.ui.btn_stereo_disparity_map.clicked.connect(self.stereo_disparity_map)


    def load_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self,
                  "Open folder",
                  "./")                 # start path
        print(self.folder_path)

    def load_image_L(self):
        self.img_L, _ = QFileDialog.getOpenFileName(
            self,
            filter='Image Files (*.png *.jpg *.jpeg *.bmp)')           # start path
        print(self.img_L)

    def load_image_R(self):
        self.img_R, _ = QFileDialog.getOpenFileName(
            self,
            filter='Image Files (*.png *.jpg *.jpeg *.bmp)')           # start path
        print(self.img_R)


    def find_corners(self):


        for i in range(1, 16):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                    cv2.drawChessboardCorners(img_src, self.pattern_size, corners2, ret)
                cv2.namedWindow("img", 0)
                cv2.resizeWindow("img", 1075, 900)
                cv2.imshow("img", img_src)
                cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def find_intrinsic(self):

        for i in range(1, 16):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    self.world_points.append(self.world_point)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                if  [corners2]:
                    self.img_points.append(corners2)
                else:
                    self.img_points.append(corners)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.world_points, self.img_points , size,None,None)
        print("instrinsic:\n", self.mtx)

    def numbers(self):
        self.number = self.ui.bmp_number.currentText()
        print(self.number)

    def find_extrinsic(self):
        file_path = ('{}/{}.bmp'.format(self.folder_path,self.number) )
        img_src = cv2.imread(file_path)

        if img_src is not None:

            gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                _, self.rvec, self.tvec, inliers = cv2.solvePnPRansac(self.world_point, corners2, self.mtx, self.dist)

                rotation_m, _ = cv2.Rodrigues(self.rvec)
                rotation_t = np.hstack([rotation_m, self.tvec])
                print("extrinsic:\n",rotation_t)

    def find_distortion(self):
        print("distortion:\n", self.dist)

    def show_result(self):
        for i in range(1, 16):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, size, 0, size)

                dst = cv2.undistort(img_src, self.mtx, self.dist, None, newcameramtx)

            imgs = np.hstack([img_src,dst])
            cv2.namedWindow("img", 0)
            cv2.resizeWindow("img", 1920, 1080)
            cv2.imshow("img", imgs)
            cv2.waitKey(1000)
        cv2.destroyAllWindows()

    def words_on_board(self):
        self.words = self.ui.words_input.toPlainText()
        world_point = np.zeros((self.corners_vertical * self.corners_horizontal, 3), np.float32)
        world_point[:, :2] = np.mgrid[:self.corners_horizontal, :self.corners_vertical].T.reshape(-1, 2)
        world_points = []
        img_points = []

        position = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]
        for i in range(1,6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    world_points.append(world_point)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, size, None, None)
        for i in range(1, 6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size)

                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, corners2, mtx,dist)

                    fs = cv2.FileStorage('{}/Q2_lib/alphabet_lib_onboard.txt'.format(self.folder_path), cv2.FILE_STORAGE_READ)
                    for j in range(len(self.words)):
                        if self.words[j] is not None:
                            w = []
                            wi = []
                            ch = fs.getNode('{}'.format(self.words[j])).mat()
                            for k in range(len(ch)):
                                w.append(ch[k][0]+position[j])
                                wi.append(ch[k][1]+position[j])
                            word = np.float32(w).reshape(-1,3)
                            imgpts, jac = cv2.projectPoints(word, rvec, tvec, mtx, dist)
                            word_i = np.float32(wi).reshape(-1,3)
                            imgpts_i =cv2.projectPoints(word_i, rvec, tvec, mtx, dist)
                            for a in range(len(ch)):
                                corners2[0][0][0] = imgpts_i[0][a][0][0]
                                corners2[0][0][1] = imgpts_i[0][a][0][1]
                                corner = tuple(corners2[0].ravel())
                                img_src = cv2.line(img_src, corner, tuple(imgpts[a].ravel()), (0, 0, 255), 5)

                    cv2.namedWindow("img", 0)
                    cv2.resizeWindow("img", 1075, 900)
                    cv2.imshow("img", img_src)
                    cv2.waitKey(1000)
                cv2.destroyAllWindows()

    def words_vertically(self):
        self.words = self.ui.words_input.toPlainText()
        world_point = np.zeros((self.corners_vertical * self.corners_horizontal, 3), np.float32)
        world_point[:, :2] = np.mgrid[:self.corners_horizontal, :self.corners_vertical].T.reshape(-1, 2)
        world_points = []
        img_points = []

        position = [[7, 5, 0],[4, 5, 0],[1, 5, 0],[7, 2, 0],[4, 2, 0],[1, 2, 0]]
        for i in range(1,6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)

                if ret:
                    world_points.append(world_point)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, size, None, None)
        for i in range(1, 6):
            file_path = ('{}/{}.bmp'.format(self.folder_path,i) )
            img_src = cv2.imread(file_path)

            if img_src is not None:

                gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                size = gray.shape[::-1]
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size)

                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, corners2, mtx,dist)

                    fs = cv2.FileStorage('{}/Q2_lib/alphabet_lib_vertical.txt'.format(self.folder_path), cv2.FILE_STORAGE_READ)
                    for j in range(len(self.words)):
                        if self.words[j] is not None:
                            w = []
                            wi = []
                            ch = fs.getNode('{}'.format(self.words[j])).mat()
                            for k in range(len(ch)):
                                w.append(ch[k][0]+position[j])
                                wi.append(ch[k][1]+position[j])


                            word = np.float32(w).reshape(-1,3)
                            imgpts, jac = cv2.projectPoints(word, rvec, tvec, mtx, dist)
                            word_i = np.float32(wi).reshape(-1,3)
                            imgpts_i =cv2.projectPoints(word_i, rvec, tvec, mtx, dist)
                            for a in range(len(ch)):
                                corners2[0][0][0] = imgpts_i[0][a][0][0]
                                corners2[0][0][1] = imgpts_i[0][a][0][1]
                                corner = tuple(corners2[0].ravel())
                                img_src = cv2.line(img_src, corner, tuple(imgpts[a].ravel()), (0, 0, 255), 5)

                    cv2.namedWindow("img", 0)
                    cv2.resizeWindow("img", 1075, 900)
                    cv2.imshow("img", img_src)
                    cv2.waitKey(1000)
                cv2.destroyAllWindows()

    def stereo_disparity_map(self):

        img_L = cv2.imread('{}'.format(self.img_L), 0)
        img_R = cv2.imread('{}'.format(self.img_R), 0)


        stereo = cv2.StereoBM_create(numDisparities=16*16, blockSize=25)
        disparity = stereo.compute(img_L,img_R)
        disparity = (disparity - np.min(disparity)) * (255 / (np.max(disparity) - np.min(disparity)))
        img_L = cv2.imread('{}'.format(self.img_L))
        img_R = cv2.imread('{}'.format(self.img_R))

        height = disparity.shape[0]
        width =disparity.shape[1]
        img_L = cv2.resize(img_L, (int(width*0.5),int(height*0.5)), interpolation=cv2.INTER_AREA)
        img_R = cv2.resize(img_R, (int(width*0.5),int(height*0.5)), interpolation=cv2.INTER_AREA)
        disparity = cv2.resize(disparity, (int(width*0.5),int(height*0.5)), interpolation=cv2.INTER_AREA)

        baseline=343
        focal_length=4019
        Cx = 279


        def draw_circle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN :
                if disparity[y][x] > 0 :
                    dist = disparity[y][x] - Cx
                    depth = int(focal_length * baseline / abs(dist))
                    cv2.circle(img_R, ( x - int( disparity[y][x] / 2 )  , y ), 5 , (0, 0, 255), -1)

                    cv2.imshow('imgr', img_R)
        while (True):
            cv2.namedWindow('imgl')
            cv2.namedWindow('imgr')
            cv2.setMouseCallback('imgl', draw_circle, None)
            cv2.imshow('imgl', img_L)
            cv2.imshow('imgr', img_R)
            plt.imshow(disparity, 'gray')
            plt.show()
            if cv2.waitKey(20)  :
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()