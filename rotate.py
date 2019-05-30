#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 15:46:00 2018

@author: walter
"""

import utils
import cv2

files = utils.takeAllFiles('/home/walter/Documents/ruttner/rotate')

for file in files:
	img = cv2.imread(file)
	rows, cols, _ = img.shape
	center = (cols/2, rows/2)
	M = cv2.getRotationMatrix2D(center, 180, 1)
	img = cv2.warpAffine(img, M, (cols, rows))
	cv2.imwrite(file,img)