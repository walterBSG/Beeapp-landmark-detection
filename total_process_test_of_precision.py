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
from utils import takeFilesByExtension, shift_angle, takeAllFiles
import cv2

import time
final = time.time()
start = time.time()

def gather_paths():
	
	folders = []
	final_paths = []
	ruttner = '/home/walter/Documents/ruttner/classes'
	#folders.append('/home/walter/Documents/Projeto_Asas/Asas_Peninsula_iberica_FEMEAS')
	#folders.append('/home/walter/Documents/Projeto_Asas/ASAS_ACORES_2017')
	
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

from utils import takeFloatsAndNames
	
# pass cutted images through the unet
def test_unets(unet_model_name):
	all_masks = []
	unet_model = unet.load_with(unet_model_name)
	for wings in total_wings:
		wings_masks = unet.use_unet(wings, unet_model)
		print(type(wings_masks[0]))
		wings_masks = shift_angle(wings_masks)
		all_masks.append(wings_masks)
		
	# detect the center of the "dots" created by the unet
	all_dots = []
	right_names = []
	for masks, names in zip(all_masks, total_names):
		print('Blob Detection')
		dots_list, names_temp = BlobDetection.take_points_with_reference(masks, names)
		dots_list = pr.list_dot_check(dots_list)
		right_names.append(names_temp)
		all_dots.append(dots_list)
	
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
	
	from scipy.spatial import procrustes
	from scipy.spatial.distance import euclidean
	
	def difference(v1, v2):
		if v1 > v2:
			return (v2/v1)
		return (v1/v2)
	
	def compareShape(old_shape, shape, reference):
		_, old_shape, _ = procrustes(reference, old_shape)
		old_distance = []
		for dot in old_shape:
			for dot_ in old_shape:
				old_distance.append(euclidean(dot,dot_))
		old_distance = [v for v in old_distance if v != 0]
		
		new_distance = []
		for dot in shape:
			for dot_ in shape:
				new_distance.append(euclidean(dot,dot_))
		new_distance = [v for v in new_distance if v != 0]
		
		answer = []
		for old, new in zip(old_distance, new_distance):
			answer.append(difference(old, new))
		
		return np.asarray(answer)
		
	
	def lookAndCompare(name, reference, old_shape, new_list):
		answer = []
		for newname, shape in new_list:
			if newname == name:
				answer.append(compareShape(old_shape, shape, reference))
		
		return np.asarray(answer)
	
	names = []
	for n_list in  right_names:
		for n in n_list:
			names.append(n)
	
	
	#start of analysis
	"""
	real_deal = []
	
	n, f = takeFloatsAndNames('/home/walter/Documents/new_data/acores_400.txt')
	f = np.reshape(f, (-1,19,2))
	f = pr.list_dot_check(f)
	for  a, b in zip(n, f):
		real_deal.append((a,b))
	n, f = takeFloatsAndNames('/home/walter/Documents/new_data/peninsula_400.txt')
	f = np.reshape(f, (-1,19,2))
	f = pr.list_dot_check(f)
	for  a, b in zip(n, f):
		real_deal.append((a,b))
	n, f = takeFloatsAndNames('/home/walter/Documents/new_data/ruttner_400.txt')
	f = np.reshape(f, (-1,19,2))
	f = pr.list_dot_check(f)
	for  a, b in zip(n, f):
		real_deal.append((a,b))
		
	X = np.reshape(X, (-1,19,2))
	new_deal = []
	for a, b in zip(names, X):
		new_deal.append((a,b))
	
	answer = []
	for n, f in real_deal:
		answer.append(lookAndCompare(n, reference, f, new_deal))
	
	answer = [x for x in answer if x != []]
	
	sums = sum(answer)/len(answer)
	sums2 = sum(sums[0])/(342)
	sums = sums[0]
	sums = sums.tolist()
	NAME = '/home/walter/Documents/Results/precision/precision.txt'
	f = open(NAME,"w+")
	f.write('Total precision: {}\n\n'.format(sums2))
	while(sums):
		point = []
		for num in range(18):
			r = sums.pop(0)
			point.append(r)
			f.write('{}\t'.format(r))
		f.write('Final: {}\n'.format(sum(point)/18))
	f.close()
	
	NAME = '/home/walter/Documents/Results/precision/precision.csv'
	f = open(NAME,"w+")
	for ans in answer:
		for point in ans[0]:
			f.write('{}\t'.format(point))
		f.write('\n')
	f.close()
	
	"""
	full_count = 0
	for dots in all_dots:
		full_count += len(dots)
	print('resultado =' + str(full_count))
	
	a = unet_model_name + ": " + str(total/full_count) + '\n'
	
	with open("precision.txt", "a") as myfile:
		myfile.write(a)
		
	index = 0
	for wings, masks in zip(total_wings, all_masks):
		for wing, mask in zip(wings, masks):
			name = '/home/walter/Documents/Results/wings/%s.jpg'%index
			index += 1
			mask = cv2.addWeighted(mask,0.6,wing,0.4,0)
			cv2.imwrite(name, wing);
			name = '/home/walter/Documents/Results/detected/%s.jpg'%index
			cv2.imwrite(name, mask);

test_unets('/home/walter/Documents/cv2/k5.h5')


































