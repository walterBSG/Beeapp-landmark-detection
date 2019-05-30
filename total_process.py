#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:48:52 2019

@author: walter
"""

import os
import wing_detector as detector
import unet_module as unet
import BlobDetection
import numpy as np
import procrustes_module as pr
from utils import takeFilesByExtension, shift_angle

import time
final = time.time()
start = time.time()

def gather_paths():
	
	folders = []
	final_paths = []
	ruttner = '/home/walter/Documents/ruttner/classes'
	folders.append('/home/walter/Documents/Projeto_Asas/Asas_Peninsula_iberica_FEMEAS')
	folders.append('/home/walter/Documents/Projeto_Asas/ASAS_ACORES_2017')
	
	for directory in os.listdir(ruttner):
		folders.append(os.path.join(ruttner, directory))
	
	classes_names = []
	for folder in folders:
		classes_names.append(os.path.basename(folder))
		paths = takeFilesByExtension(folder, '*.jpg')
		paths += takeFilesByExtension(folder, '*.bmp')
		paths += takeFilesByExtension(folder, '*.JPG')
		final_paths.append(paths)
	
	return final_paths, classes_names
	
total_paths, classes_names = gather_paths()
	
# detect and cut wings
total_wings = []
total_names = []
for paths in total_paths:
	wings, names = detector.detectFromPaths(paths)
	total_wings.append(wings)
	total_names.append(names)

allpathsNumber = 0
for paths in total_paths:
	allpathsNumber += len(paths)

allWingsNumber = 0
for wings in total_wings:
	allWingsNumber += len(wings)
	
print(allpathsNumber, 'imagens encontradas, ', allWingsNumber, 'imagens foram detectadas e recortadas. Isso demorou:', time.time() - start, "segundos.")
start = time.time()
	
# pass cutted images through the unet
all_masks = []
unet_model_name = '/home/walter/Documents/cv2/raio-4-plateau-dust-pesos-kernel5.h5'
unet_model = unet.load_with(unet_model_name)
for wings in total_wings:
	wings_masks = unet.use_unet(wings, unet_model)
	print(type(wings_masks[0]))
	wings_masks = shift_angle(wings_masks)
	all_masks.append(wings_masks)
	
print('Isso demorou:', time.time() - start, "segundos.")
start = time.time()

# detect the center of the "dots" created by the unet
all_dots = []
for masks in all_masks:
	print('Blob Detection')
	dots_list = BlobDetection.take_points(masks)
	dots_list = pr.list_dot_check(dots_list)
	all_dots.append(dots_list)
	
print('Isso demorou:', time.time() - start, "segundos.")
start = time.time()

#creatre the SVM input
X = []
y = []

for index, dots_list in enumerate(all_dots): 
	for keys in dots_list:
		X.append(keys)
		y.append(index)

#X = pr.list_euclidean(X)
X, reference = pr.procrustes_analysis(X, 0.01)

def ajust_reference(X, reference):
	X = np.reshape(X, (-1, 38))
	reference = np.reshape(reference, (-1, 38))
	
	result = []
	for x in X:
		number_list = []
		for index, ref in enumerate(reference):
			number_list.append(x[index] - ref)
		result.append(number_list)
	return result
		
X = ajust_reference(X, reference)
X = np.reshape(X, (-1, 38))
	
total = 0
for index, dots_list in enumerate(all_dots):
	NAME = '/home/walter/Documents/Results/precision/new/Class_{}.tps'.format(index)
	f = open(NAME,"w+")
	for keys in dots_list:
		total += 1
		f.write('LM=19\n')
		for key in keys:
			f.write('{} {} \n'.format(float(key[0]),float(key[1])))
	
		f.write('ID=wing_{}\n\n'.format(total))
	f.close()
	
print(total)

NAME = '/home/walter/Documents/Results/precision/new/Class_Names.txt'
f = open(NAME,"w+")

for name in classes_names:
	f.write('{} \n'.format(name))

f.close()

print('Isso demorou no total:', time.time() - final, "segundos.")

NAME = '/home/walter/Documents/Results/precision/new/detections_no_procrustes.csv'
f = open(NAME,"w+")
for index, dots_list in enumerate(all_dots):
	for keys in dots_list:
		for key in keys:
			f.write('{}\t{}\t'.format(float(key[0]),float(key[1])))
		f.write('{}\n'.format(index))
f.close()

X = np.reshape(X, (-1, 38))
NAME = '/home/walter/Documents/Results/precision/new/detections_with_procrustes.csv'
f = open(NAME,"w+")
for class_, elements in zip(y, X):
	for ele in elements:
		f.write('{}\t'.format(float(ele)))
	f.write('{}\n'.format(class_))

f.close()

#create full reference
NAME = '/home/walter/Documents/Results/precision/new/main-reference.txt'
f = open(NAME,"w+")
for point in reference:
	f.write('{} {} \n'.format(float(point[0]),float(point[1])))
f.close()

#create classes reference
for index, dots_list in enumerate(all_dots):
	NAME = '/home/walter/Documents/Results/precision/new/reference_{}.tps'.format(index)
	_, new_ref = pr.procrustes_analysis(dots_list, 0.01)
	f = open(NAME,"w+")
	for point in new_ref:
		f.write('{} {} \n'.format(float(point[0]),float(point[1])))
	f.close()

full_count = 0
for masks in all_masks:
	full_count += len(masks)

a = unet_model_name + ": " + str(total/full_count) + '\n'

with open("precision.txt", "a") as myfile:
	myfile.write(a)
    
print(new_ref)















































