#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:40:00 2018

@author: walter
"""
import cv2
import wing_detector as detector
import unet_module as unet
import BlobDetection
import procrustes_module as pr
from utils import takeFilesByExtension, shift_angle

folder = '/home/walter/Documents/Projeto_Asas/Asas_Peninsula_iberica_FEMEAS'
NAME = '/home/walter/Documents/ruttner/peninsula_detected.txt'

paths = takeFilesByExtension(folder, '*.jpg')
paths = paths +  takeFilesByExtension(folder, '*.bmp')
paths = paths +  takeFilesByExtension(folder, '*.JPG')
	
# detect and cut wings
wings, names = detector.detectFromPaths(paths)

print(len(paths), 'imagens encontradas, ', len(wings), 'imagens foram detectadas e recortadas')

# saving cutted images to be evaluated
for index, wing in enumerate(wings):
	name = '/home/walter/Documents/Results/boxes/%s.jpg'%index
	cv2.imwrite(name, wing)
	
# pass cutted images through the unet
wings_masks = unet.predict_with(wings, '/home/walter/Documents/cv2/raio-4-plateau-dust-pesos-kernel5.h5')

# saving unet output to be evaluated to be evaluated
for index, mask in enumerate(wings_masks):
	name = '/home/walter/Documents/Results/detected/%s.jpg'%index
	mask = cv2.addWeighted(mask,0.6,wings[index],0.4,0)
	cv2.imwrite(name, mask);

wings_masks = shift_angle(wings_masks)

# detect the center of the "dots" created by the unet
dots_list = BlobDetection.take_points(wings_masks)

dots_list = pr.list_dot_check(dots_list)

counter = 0
f = open(NAME,"w+")
for name, keys in zip(names, dots_list):
	
	for key in keys:
		f.write('{} {} \n'.format(float(key[0]),float(key[1])))

	f.write(name + '\n')

f.close()
print(len(dots_list))
