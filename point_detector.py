#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:27:29 2018

@author: walter
"""

import wing_detector as detector
import unet_module as unet
import BlobDetection
import cv2
import numpy as np

wings = detector.detectFromFolder('/home/walter/Documents/python/adaptive-style-transfer-master/saved')
wings_masks = unet.predict_with(wings, '/home/walter/Documents/cv2/model-circulo-tamanho3_plateau.h5')
keys_list = BlobDetection.take_points(wings_masks)

for n, img in enumerate(wings_masks):
	name = '/home/walter/Documents/Results/detected/%s.jpg'%n
	img = cv2.addWeighted(img,0.6,wings[n],0.4,0)
	cv2.imwrite(name, img)
	

print(keys_list)

i = np.zeros((400, 400, 1), np.uint8)
i = cv2.drawKeypoints(i,keys_list[0], np.array([]))
cv2.imshow('img', i)
cv2.waitKey(0)
