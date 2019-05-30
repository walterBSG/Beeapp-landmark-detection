#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:22:23 2018

@author: walter
"""
import cv2
import numpy as np

def create_blob_detector(roi_size=(128, 128), blob_min_area=1, blob_min_int=.1, blob_max_int=.99, blob_th_step=1):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = blob_min_area
    params.maxArea = roi_size[0]*roi_size[1]
    params.filterByCircularity = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # blob detection only works with "uint8" images.
    params.minThreshold = int(blob_min_int*255)
    params.maxThreshold = int(blob_max_int*255)
    params.thresholdStep = blob_th_step
    return cv2.SimpleBlobDetector_create(params)

def take_points(images):
		
	thresh = 210
	maxValue = 255
	detector = create_blob_detector()
	dots_list = []
	kernel = np.ones((5,5),np.uint8)
	
	for img in images:
	
		_, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);

		keypoints = detector.detect(img)
		
		length, dots = takeDots(keypoints)
		
		if length != 19:
			img = cv2.dilate(img,kernel,iterations = 1)
			keypoints = detector.detect(img)
			length, dots = takeDots(keypoints)
		
		if length != 19:
			continue
		
		dots_list.append(dots)
	
	return dots_list

def take_points_with_reference(images, references):
		
	thresh = 210
	maxValue = 255
	detector = create_blob_detector()
	dots_list = []
	names = []
	kernel = np.ones((5,5),np.uint8)
	
	for img, reference in zip(images, references):
	
		_, img = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);

		keypoints = detector.detect(img)
		
		length, dots = takeDots(keypoints)
		
		if length != 19:
			img = cv2.dilate(img,kernel,iterations = 1)
			keypoints = detector.detect(img)
			length, dots = takeDots(keypoints)
		
		if length != 19:
			continue
		
		dots_list.append(dots)
		names.append(reference)
	
	return dots_list, names

def compare(result, original):
	
	thresh = 210
	maxValue = 255
	detector = create_blob_detector()
	dots_list = []
	kernel = np.ones((5,5),np.uint8)
	
	for res, orig in zip(result, original):
		_, res = cv2.threshold(res, thresh, maxValue, cv2.THRESH_BINARY);
		_, orig = cv2.threshold(orig, thresh, maxValue, cv2.THRESH_BINARY);
		
		keypoints_res = detector.detect(res)
		keypoints_orig = detector.detect(orig)
		
		length_res, dots_res = takeDots(keypoints_res)
		length_orig, dots_orig = takeDots(keypoints_orig)
		
		if length_res != 19:
			res = cv2.dilate(res,kernel,iterations = 1)
			keypoints_res = detector.detect(res)
			length_res, dots_res = takeDots(keypoints_res)
		
		if length_res != 19:
			continue
		
		if length_orig != 19:
			orig = cv2.dilate(orig,kernel,iterations = 1)
			keypoints_orig = detector.detect(orig)
			length_orig, dots_orig = takeDots(keypoints_orig)
		
		if length_orig != 19:
			continue
		
		dots_list.append([dots_res, dots_orig])
	
	return dots_list
		
def takeDots(key):
	points = []
	for point in key:
		points.append((int(point.pt[0]),int(point.pt[1])))
	
	return len(points), points

##########################################################################################################################