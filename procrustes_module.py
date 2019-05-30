#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:42:41 2019

@author: walter
"""

from random import randint
from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean
from numpy import linalg as LA
import numpy as np

def aligh_shapes_euclidean(obj, reference):
	"""
	compare similar objects and say witch point of obj2
	corresponds to another in obj1 using distance as criteria
	"""
	_, obj, _ = procrustes(reference, obj)
	obj = list(obj)
	result = []
	
	for point in reference:
		distance = euclidean(obj[0], point)
		target = 0
		
		for index, p in enumerate(obj):
			d = euclidean(p, point)
			if distance > d:
				distance = d
				target = index
		
		result.append(obj.pop(target))
		
	return result

def list_euclidean(shapes):
	results = []
	
	shape = list_dot_check(shapes)
	
	shapes, reference = procrustes_analysis(shapes, 0.01)
	
	for shape in shapes:
		results.append(aligh_shapes_euclidean(shape, reference))
	
	return results

def dot_check(data, difference = 5):
	
	data = sorted(data, key=lambda d: d[0])
	result = []
	while(data):
		result, data = distance_check(result, data, difference)
		
	return result

def distance_check(result, data, difference = 5):
	newPiece = [data.pop(0)]
	for d in data:
		if (d[0] - difference) <= newPiece[0][0]:
			newPiece.append(data.pop(0))
		else:
			break
	newPiece = sorted(newPiece, key=lambda r: r[1])
	result += newPiece
	return result, data

def list_dot_check(data, difference = 5):
	result = []
	for d in data:
		element = dot_check(d,difference)
		element = adjust_element(element)
		result.append(element)
		
	return result

def adjust_element(element):
	#test problematic areas
	element[3:8] = sorted(element[3:8], key=lambda r: r[1])
	element[9:11] = sorted(element[9:11], key=lambda r: r[1])
	element[13:15] = sorted(element[13:15], key=lambda r: r[1])
	e1 = element[5]
	e2 = element[8]
	if e1[1] < e2[1]:
		element[5] = e2
		element[8] = e1
	e1 = element[13]
	e2 = element[15]
	if e1[1] > e2[1]:
		element[13] = e2
		element[15] = e1
	e1 = element[0]
	e2 = element[1]
	if e1[1] > e2[1]:
		element[0] = e2
		element[1] = e1
	return element

def region_check(data):
	"""
	patternize the order of point in a geometric object based on its position
	"""
	
	from sklearn import preprocessing
	data = preprocessing.normalize(data)
	start = 0
	end = 0.1
	result = []
	data = sorted(data, key=lambda d: d[0])
	while(end<=1):
		test = []
		for d in data:
			if start <= d[0] and d[0] <= end:
				test.append(d)
				data.remove(d)
		
		test = sorted(test, key=lambda t: t[1])
		result += test
		start = end
		end += 0.1
	return result

def kmeans(data, clusters):
	from sklearn.cluster import KMeans	
	
	amount = clusters*len(data)
	data = np.reshape(data, (amount, 2))
	
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)
	return kmeans

def compare(ref, mean, t):
	disparity = LA.norm(np.subtract(ref, mean))
	
	if disparity > t:
		print('not quite:{}'.format(disparity))
		return False, mean
	return True, ref

def procrustes_analysis(data, threashold):
	
	reference = data[randint(0,len(data)-1)]
	print(reference)
	
	while (True):
		fited = []
		for d in data:
			_, m2, _ = procrustes(reference, d)
			fited.append(m2)
		
		mean_shape = np.average(fited, axis=0)
		boolean, reference = compare(reference,mean_shape, threashold)
		
		if boolean:
			break
	
	return fited, reference

def takeFloats(file):
	
	with open(file, 'r') as f:
		data = f.read().split()
		floats = []
		for elem in data:
			try:
				floats.append(float(elem))
			except ValueError:
				pass
				
	return floats

def TakeAll():
	f = takeFloats('/home/walter/Documents/ruttner/acores_detected.txt')
	f += takeFloats('/home/walter/Documents/ruttner/peninsula_detected.txt')
	f += takeFloats('/home/walter/Documents/ruttner/ruttner_detected.txt')
	
	result = []
	while(f):
		obj = []
		for i in range(19):
			obj.append([int(f.pop(0)),int(f.pop(0))])
	
		result.append(obj)
		
	return result

def compare_references(file1, file2):
	import utils
	_, reference1 = utils.takeFloatsAndNames(file1)
	reference1 = np.reshape(reference1,(-1,2))
	_, reference2 = utils.takeFloatsAndNames(file2)
	reference2 = np.reshape(reference2,(-1,2))
	_, _, d = procrustes(reference1, reference2)
	
	return d
	
def compare_lists(list1, list2):
	_, reference = procrustes_analysis(list1, 0.01)
	result = []
	for obj in list2:
		_,_,d = procrustes(reference, obj)
		result.append(d)
	return result
"""
import utils
import math
_, file1 = utils.takeFloatsAndNames('/home/walter/Documents/ruttner/Class_1.tps')
_, file2 = utils.takeFloatsAndNames('/home/walter/Documents/Projeto_Asas/acores_correto.txt')
file1 = np.reshape(file1,(-1,19,2))
file2 = np.reshape(file2,(-1,19,2))
result =  compare_lists(file1, file2)
result = np.asarray(result)
a = math.sqrt(result.mean())
"""















































