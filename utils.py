#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:04:55 2018

@author: walter
"""
import os
import shutil
from PIL import Image
from fnmatch import fnmatch
import numpy as np
import cv2
from sklearn.decomposition import PCA

def takeNamesAndFloats(file, folder):
	names = []
	
	with open(file, 'r') as f:
		data = f.read().split()
		floats = []
		for elem in data:
			try:
				floats.append(float(elem))
			except ValueError:
				pass
	
	with open(file, 'r') as f:
		for line in f:
			if '.jpg' in line or '.bmp' in line or '.JPG' in line:
				if '.JPG' in line:
					line = os.path.join(file,line.replace('.JPG','.jpg'))
				names.append(os.path.join(folder,line.replace('\n','')))
	return names, floats

def takeFloatsAndNames(file):
	names = []
	
	with open(file, 'r') as f:
		data = f.read().split()
		floats = []
		for elem in data:
			try:
				floats.append(float(elem))
			except ValueError:
				pass
	
	with open(file, 'r') as f:
		for line in f:
			if '.jpg' in line or '.bmp' in line or '.JPG' in line:
				if '.JPG' in line:
					line = os.path.join(file,line.replace('.JPG','.jpg'))
				names.append(line.replace('\n',''))
				
	return names, floats

def takeFilesByExtension(folder, pattern):
	paths = []
	for path, subdirs, files in os.walk(folder):
	    for name in files:
	        if fnmatch(name, pattern):
	            paths.append(os.path.join(path, name))

	return paths

def takeAllFiles(folder):
	paths = []
	for path, subdirs, files in os.walk(folder):
		for name in files:
			paths.append(os.path.join(path, name))
	return paths

def list_paths(folder):
	names = []
	for filename in os.listdir(folder):
		filename = os.path.join(folder,filename)
		names.append(filename)
	names = sorted(names, key=str.lower)
	return names

def findNames(file):
	names = []
	with open(file) as f:
		for line in f:
			if "ID=" in line:
				line = line.replace('ID=','')
				line = line.replace('\n','')
				names.append(line)
	return names

def loadImages(paths):
	images = []
	for path in paths:
		images.append(Image.open(path))
	
	return images

def copy(folder,dest):
	
	files = os.listdir(folder)
	files = sorted(files, key=str.lower)
	
	for idx, n in enumerate(files):
		if not (os.path.isdir(folder+n)):
			shutil.copyfile(os.path.join(folder,n), os.path.join(dest,n))
			
def copyByPaths(paths,dest):	
	for path in paths:
		name = os.path.basename(path)
		shutil.copyfile(path, os.path.join(dest,name))

def move_random_percentage(folder,dest, percentage):
	from random import randint
	files = os.listdir(folder)
	amount = int((percentage/100)*len(files))
	for _ in range(amount):
		name = files.pop(randint(0,len(files)-1))
		shutil.move(os.path.join(folder,name), os.path.join(dest,name))

def move_percentage_bySteps(folder,dest, percentage,step):
	files = os.listdir(folder)
	amount = int((percentage/100)*len(files))
	counter = 0
	for _ in range(amount):
		try:
			name = files.pop(counter)
			counter += step
			shutil.move(os.path.join(folder,name), os.path.join(dest,name))
		except:
			counter = 0
			name = files.pop(counter)
			counter += step
			shutil.move(os.path.join(folder,name), os.path.join(dest,name))
			
def remove(original, target):
	files = os.listdir(original)
	for file in files:
		   os.remove(os.path.join(target,file))
   
def clean(folder):
	files = os.listdir(folder)
	for file in files:
		os.remove(os.path.join(folder,file))   

def shift_angle(imgs):
	
	for index, img in enumerate(imgs):
		
		rows, cols = img.shape
		center = (cols/2,rows/2)
		_, img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY);
		
		dots = np.argwhere(img == 255)
		if len(dots) == 0:
			continue
		
		pca = PCA(n_components=2)
		pca.fit_transform(dots)
		
		vec = pca.components_
		angle = np.degrees(np.arctan2(*vec.T[::-1])) % 360.0
		
		#move img to origin
		M = np.float32([[1,0,-(pca.mean_[1]-center[1])],[0,1,-(pca.mean_[0]-center[0])]])
		img = cv2.warpAffine(img,M,(cols,rows))
		
		#rotate img
		M = cv2.getRotationMatrix2D((center),-angle[1],1)
		img = cv2.warpAffine(img,M,(cols,rows))
				
		imgs[index] = img
	
	return imgs

def rotate_dots(dots, center, angle):
	theta = np.radians(angle)
	c, s = np.cos(theta), np.sin(theta)
	R = np.array(((c,-s), (s, c)))
	result = []
	for dot in dots:
		dot = [dot[0]-center[0], dot[1]-center[1]]
		dot = np.dot(R,dot)
		result.append([dot[0]+center[0], dot[1]+center[1]])
	
	return result

def list_to_file(list_, names, file_name):
	
	f = open(file_name,'w+')
	
	for name, obj in zip(names,list_):
		for dot in obj:
			f.write('{} {} \n'.format(dot[0],dot[1]))
		
		f.write(name + '\n')
	f.close



































