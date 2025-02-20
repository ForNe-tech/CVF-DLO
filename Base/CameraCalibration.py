# -*- coding: utf-8 -*-
import random

import cv2
import numpy as np
import glob


class Cphoto_pre_work:

    def __init__(self):
        self.obj_points = []  # 存储3D点
        self.img_points = []  # 存储2D点
        self.size = []
        self.ret = []
        self.mtx = []
        self.dist = []
        self.rvecs = []
        self.tvecs = []
        self.list1 = ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']
        self.savesrc5 = []

    def get_grid(self, src, src_corner):
        # 进行标定
        # src:图像路径
        # src_corner:棋盘格坐标路径
        images = glob.glob(src)
        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

        # 获取标定板角点的位置
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
        i = 0
        random.shuffle(images)
        images_s = images
        for fname in images_s:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret:
                self.obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点

                if [corners2]:
                    self.img_points.append(corners2)
                else:
                    self.img_points.append(corners)
                i += 1
                # cv2.drawChessboardCorners(img, (11, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
                # cv2.imwrite(src_corner + fname.rsplit("\\", 1)[1], img)
                # cv2.imwrite('./outimg/conimg'+str(i)+'.jpg', img)
                # cv2.waitKey(4000)
        if self.img_points:
            return True
        else:
            return False

    def get_parameter5(self, savesrc):
        # 保存相应5个标定参数，并存储
        self.savesrc5 = savesrc
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                                                    self.size, None, None)
        np.save(savesrc + self.list1[0], self.ret)
        np.save(savesrc + self.list1[1], self.mtx)
        np.save(savesrc + self.list1[2], self.dist)
        np.save(savesrc + self.list1[3], self.rvecs)
        np.save(savesrc + self.list1[4], self.tvecs)

    def last_photo(self, imgsrc, src_par5, last_src):
        # imgsrc:原始影像路径
        # src_part5:5参数路径
        # last_src:最后保存路径
        # 选择指定文件夹即可直接读入文件夹内保存的.npy数据,生成相片
        origin_images = glob.glob(imgsrc + '\*.[jp][pn]g')
        for tempfname in origin_images:
            tempimg = cv2.imread(tempfname)
            h, w = tempimg.shape[:2]
            # print(tempfname.rsplit("\\",1)[1])
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.load(src_par5 + self.list1[1] + '.npy'),
                                                              np.load(src_par5 + self.list1[2] + '.npy'), (w, h), 1,
                                                              (w, h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
            print(newcameramtx)
            print("------------------使用undistort函数-------------------")
            temp_dst = cv2.undistort(tempimg, np.load(src_par5 + self.list1[1] + '.npy'),
                                     np.load(src_par5 + self.list1[2] + '.npy'), None, newcameramtx)
            x, y, w, h = roi
            tempdst1 = temp_dst[y:y + h, x:x + w]
            cv2.imwrite(last_src + tempfname.rsplit("\\", 1)[1], tempdst1)
            print("方法一:dst的大小为:", tempdst1.shape)


if __name__ == '__main__':
    temp = Cphoto_pre_work()
    temp.get_grid('../../../TEMP/CamCalib/pic_s/*.jpg', '../../../TEMP/CamCalib/src_corner/')
    srccc = '../../../TEMP/CamCalib/para/'
    temp.get_parameter5(srccc)
    # temp.last_photo('../../../TEMP/CamCalib/pic/', srccc, '../../../TEMP/CamCalib/saveimg' + '/')

    file = np.load('../../../../TEMP/CamCalib/para/mtx.npy')
    np.savetxt('../../../TEMP/CamCalib/para/mtx.txt', file)