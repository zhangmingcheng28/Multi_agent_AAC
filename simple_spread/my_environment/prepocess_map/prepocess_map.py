#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：multi-agent RL
@File ：prepocess_map.py
@Author ：moshuilanting
@Date : 2023/11/17
"""

import cv2
import numpy as np
from PIL import Image
import random

if __name__=="__main__":
    # '''
    # 获取蓝色区域
    # '''
    # # 加载地图
    # image = cv2.imread("small_map.png")
    #
    # # 将图片转换为HSV颜色空间
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # # 设置蓝色的HSV范围
    # lower_blue = np.array([110, 50, 50])
    # upper_blue = np.array([130, 255, 255])
    # # 根据HSV范围创建掩码
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask[mask > 0] = 255
    #
    # # # 对原始图像应用掩码
    # # blue_area = cv2.bitwise_and(image, image, mask=mask)
    #
    # # 显示原始图像和筛选出的蓝色区域
    # #cv2.imshow("Original Image", image)
    # cv2.imshow("Blue Area", mask)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()
    #
    # cv2.imwrite('bule_area.jpg',mask)

    # '''
    # 填补区域缝隙
    # '''
    # image = cv2.imread('bule_area.jpg', cv2.IMREAD_GRAYSCALE)
    # _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # kernel_size = 15
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # filled_image = cv2.bitwise_not(closed_image)
    # cv2.imwrite('output_image.jpg', filled_image)


    '''
    轮廓检测
    '''
    image = cv2.imread('output_image.jpg')
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # 二值化图像，将灰度图像转换为黑白图像
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    # 寻找黑色区域的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图上绘制轮廓
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    # 显示图片
    cv2.imwrite('output_image.jpg', image)
    cv2.imshow('Contours', image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    # '''
    # 随机生成点
    # '''
    # img = cv2.imread('output_image.jpg',0)
    # # 二值化处理，将灰度图转换为二进制图像
    # _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # # 获取白色像素的坐标
    # white_pixels = np.where(binary_img == 255)
    # white_pixel_coords = list(zip(white_pixels[1], white_pixels[0]))
    # # 随机选择一个白色像素的坐标
    # random_pixel = random.choice(white_pixel_coords)
    # x, y = random_pixel
    # print('随机生成的点坐标：', x, y)
    # #在原图上标记生成的点
    # cv2.circle(img, (x, y), 3, (0, 0, 125), -1)
    # cv2.imshow('Random Point', img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
