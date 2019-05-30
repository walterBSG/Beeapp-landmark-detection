#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:52:02 2018

@author: walter
"""

import cv2 as cv

graph = '/home/walter/Documents/models/research/object_detection/mobile_1/frozen_inference_graph.pb'
graphTxt = '/home/walter/Documents/models/research/object_detection/itens.pbtxt'
cvNet = cv.dnn.readNetFromTensorflow(graph, graphTxt)

img = cv.imread('/home/walter/Documents/ruttner/test/3.jpg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', img)
cv.waitKey()