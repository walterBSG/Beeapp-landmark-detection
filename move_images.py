#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:26:46 2018

@author: walter
"""

import os
import cria_mascara as cm
from PIL import Image, ImageFilter
from random import randint
import numpy as np
import cv2
import utils

def data_augmentation(train,masks, amount):
	files = os.listdir(train)
	for file in files:
		img = Image.open(os.path.join(train,file))
		mask = Image.open(os.path.join(masks,file))
		for i in range(amount):
			x = randint(-20,20)
			y = randint(-80,80)
			rotation = randint(-70,70)
			newImg = img.rotate(rotation)
			newMask = mask.rotate(rotation)
			newImg = newImg.transform(img.size, Image.AFFINE, (1, 0, x, 0, 1, y))
			newMask = newMask.transform(img.size, Image.AFFINE, (1, 0, x, 0, 1, y))
			newFinal = '%s.jpg'%i
			name = file.replace('.jpg',newFinal)
			newImg.save(train + name)
			newMask.save(masks + name)
	
	utils.move_random_percentage(train, '/home/walter/Documents/safe', 60)

	addNoise(train)
	addDust(train)
	
	utils.move_random_percentage('/home/walter/Documents/safe', train , 100)

def data_landmark_augmentation(amount):
	
	names, floats =  utils.takeNamesAndFloats('/home/walter/Documents/Projeto_Asas/acores_full_crop_400.txt','/home/walter/Documents/Projeto_Asas/train')
	n, f = utils.takeNamesAndFloats('/home/walter/Documents/Projeto_Asas/peninsula_full_crop_400.txt','/home/walter/Documents/Projeto_Asas/train')
	names += n
	floats += f
	n, f = utils.takeNamesAndFloats('/home/walter/Documents/ruttner/ruttnet_400.txt','/home/walter/Documents/Projeto_Asas/train')
	names += n
	floats += f
	floats = np.reshape(floats,(-1,19,2))
	dots = []
	new_names = []
	for name, obj in zip(names,floats):
		img = cv2.imread(name)
		if img is not None:
			dots.append(obj)
			new_names.append(name)
			for i in range(amount):
				#get random values
				x = randint(-20,20)
				y = randint(-80,80)
				rotation = randint(-70,70)
				#create varibleas
				rows,cols,_ = img.shape
				M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1)
				#rotation
				newImg = cv2.warpAffine(img,M,(cols,rows))
				new_obj = utils.rotate_dots(obj,[cols/2,rows/2],rotation)
				#translation
				M = np.float32([[1,0,x],[0,1,y]])
				newImg = cv2.warpAffine(newImg,M,(cols,rows))
				new_obj = [[point[0]-x, point[1]-y] for point in new_obj]
				#save img
				newFinal = '%s.jpg'%i
				name = name.replace('.jpg',newFinal)
				cv2.imwrite(name,newImg)
				dots.append(new_obj)
				new_names.append(name)
	
	utils.list_to_file(dots, new_names, '/home/walter/Documents/Projeto_Asas/dots_augmentation.txt')
	
	utils.move_percentage('/home/walter/Documents/Projeto_Asas/train', '/home/walter/Documents/safe', 70)

	addNoise('/home/walter/Documents/Projeto_Asas/train')
	addDust('/home/walter/Documents/Projeto_Asas/train')
	
	utils.move_percentage('/home/walter/Documents/safe', '/home/walter/Documents/Projeto_Asas/train' , 100)

def addNoise(folder):
	files = os.listdir(folder)
	for file in files:
		name =os.path.join(folder, file)
		img = cv2.imread(name)
		img = addOneNoise('gauss',img)
		cv2.imwrite(name, img)

def addDust(folder):
	files = os.listdir(folder)
	for file in files:
		name =os.path.join(folder, file)
		img = Image.open(name)
		img = img.filter(ImageFilter.SHARPEN)
		dustName = "/home/walter/Pictures/{}.png".format(str(randint(1,2)))
		dust = Image.open(dustName)
		
		#define random square of dust
		width, height = dust.size
		square = randint(400,999)
		Uy = randint(square,height)
		Ux = randint(square,width)
		By = Uy - square
		Bx = Ux - square
		dust = dust.crop((Bx, By, Ux, Uy))
		#dust.putalpha(randint(0,255))
		dust = dust.resize((400,400), Image.ANTIALIAS)
		
		img.paste(dust, (0, 0), dust)
		img.save(name)

def addOneNoise(noise_typ,image):
	if noise_typ == "gauss":
		row,col,ch = image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy
	elif noise_typ == "s&p":
		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
				for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
				for i in image.shape]
		out[coords] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = image + image * gauss
		return noisy

def prepare_wings_detection():
	#######################################################################
	#make XMLs
	import criaXmlCompleto as xml
	xml.makeXMLs()
	
	#copy dataset to images folder
	folder = '/home/walter/Documents/Projeto_Asas/ASAS_ACORES_2017/'
	dest = '/home/walter/Pictures/images/'
	utils.copy(folder,dest)

	folder = '/home/walter/Documents/ruttner/ruttnerCatalogado/'
	utils.copy(folder,dest)
	
	folder = '/home/walter/Documents/Projeto_Asas/Asas_Peninsula_iberica_FEMEAS/'
	utils.copy(folder,dest)
	
	folder = dest
	dest = '/home/walter/Documents/models/research/object_detection/data/images/'
	utils.copy(folder,dest)
	
	folder = '/home/walter/Documents/Projeto_Asas/XML/'
	utils.copy(folder,dest)
	
	folder = '/home/walter/Documents/models/research/object_detection/data/images/'
	dest = '/home/walter/Documents/models/research/object_detection/data/images/train/'
	utils.copy(folder,dest)
			
			########################################################################################################
			# take 20% of dataset to test (validation)
	
	folder = '/home/walter/Pictures/images/'
	dest = '/home/walter/Documents/models/research/object_detection/data/images/test/'
	utils.move_percentage(folder, dest, 20)
	
	folder = '/home/walter/Documents/Projeto_Asas/XML/'
	utils.move_percentage(folder, dest, 20)
	
	        ####################################################
			#take another 20% of the dataset to true_test
			
	folder = '/home/walter/Pictures/images/'
	dest = '/home/walter/Documents/models/research/object_detection/data/images/true_test/'
	utils.move_percentage(folder, dest, 25)
	
	folder = '/home/walter/Documents/Projeto_Asas/XML/'
	utils.move_percentage(folder, dest, 25)
	
			####################################################
	
	##remove files from train
	
	original = '/home/walter/Documents/models/research/object_detection/data/images/test/'
	target = '/home/walter/Documents/models/research/object_detection/data/images/train/'
	utils.remove(original,target)
	
	
	original = '/home/walter/Documents/models/research/object_detection/data/images/true_test/'
	try:
		utils.remove(original, target)
	except:
		print('ups')

def prepare_unet(aug, circle):
	
	utils.clean('/home/walter/Documents/Projeto_Asas/masks/')
	utils.clean('/home/walter/Documents/Projeto_Asas/train/')
	utils.clean('/home/walter/Documents/Projeto_Asas/test/')
	utils.clean('/home/walter/Documents/Projeto_Asas/test_masks/')
	
	#move images to train
	folder = '/home/walter/Documents/Projeto_Asas/acores_square/'
	dest = '/home/walter/Documents/Projeto_Asas/train/'
	utils.copy(folder,dest)
	folder = '/home/walter/Documents/Projeto_Asas/peninsula_square/'
	utils.copy(folder,dest)
	folder = '/home/walter/Documents/ruttner/square/'
	utils.copy(folder,dest)	
	
	#make masks
	folder = '/home/walter/Documents/Projeto_Asas/train/'
	dest = '/home/walter/Documents/Projeto_Asas/masks/'
	cm.create_masks(folder, dest, circle)
	
	#move images and masks to test
	dest = '/home/walter/Documents/Projeto_Asas/test/'
	utils.move_percentage_bySteps(folder, dest, 20,5) # here is the bug
	
	folder = '/home/walter/Documents/Projeto_Asas/masks/'
	dest = '/home/walter/Documents/Projeto_Asas/test_masks/'
	utils.move_percentage_bySteps(folder, dest, 20,5) # here is the bug
	
	data_augmentation('/home/walter/Documents/Projeto_Asas/train/', '/home/walter/Documents/Projeto_Asas/masks/', aug)
	data_augmentation('/home/walter/Documents/Projeto_Asas/test/', '/home/walter/Documents/Projeto_Asas/test_masks/',2)
	
def prepare_landmark_detector():
	
	utils.clean('/home/walter/Documents/Projeto_Asas/masks/')
	utils.clean('/home/walter/Documents/Projeto_Asas/train/')
	utils.clean('/home/walter/Documents/Projeto_Asas/test/')
	utils.clean('/home/walter/Documents/Projeto_Asas/test_masks/')
	
	#move images to train
	folder = '/home/walter/Documents/Projeto_Asas/acores_square/'
	dest = '/home/walter/Documents/Projeto_Asas/train/'
	utils.copy(folder,dest)
	folder = '/home/walter/Documents/Projeto_Asas/peninsula_square/'
	utils.copy(folder,dest)
	folder = '/home/walter/Documents/ruttner/square/'
	utils.copy(folder,dest)	
	
	data_landmark_augmentation(4)


def clahe():
	
	claheFilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	
	for directory in ['/home/walter/Documents/Projeto_Asas/train','/home/walter/Documents/Projeto_Asas/test']:
		image_path = os.path.join(os.getcwd(), '{}'.format(directory))
		img = cv2.imread(image_path)
		img = claheFilter.apply(img)
		cv2.imwrite(image_path, img)
	
def prepareClassifier():
	import shutil

	folder = '/home/walter/Documents/ruttner/Asas_Ruttner'
	paths = utils.takeFilesByExtension(folder, '*.jpg')
	paths = paths + utils.takeFilesByExtension(folder, '*.bmp')
	
	dest = '/home/walter/Documents/ruttner/classes'
	print(paths[1])
	for path in paths:
		if '-1.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.mellifera',os.path.basename(path)))
		if '-2.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.iberica',os.path.basename(path)))
		if '-3.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.major',os.path.basename(path)))
		if '-4.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.intermissa',os.path.basename(path)))
		if '-5.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.sahariensis',os.path.basename(path)))
		if '-6.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.carnica',os.path.basename(path)))
		if '-7.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.macedonica',os.path.basename(path)))
		if '-8.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.cecropia',os.path.basename(path)))
		if '-9.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.ligustica',os.path.basename(path)))
		if '-10.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.sicula',os.path.basename(path)))
		if '-11.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.caucasica',os.path.basename(path)))
		if '-12.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.armeniaca',os.path.basename(path)))
		if '-13.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.meda',os.path.basename(path)))
		if '-14.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.anatoliaca',os.path.basename(path)))
		if '-15.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.syriaca',os.path.basename(path)))
		if '-16.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.cypria',os.path.basename(path)))
		if '-17.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.adami',os.path.basename(path)))
		if '-18.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.lamarkii',os.path.basename(path)))
		if '-19.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.jemenitica',os.path.basename(path)))
		if '-20.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.litorea',os.path.basename(path)))
		if '-21.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.scutellata',os.path.basename(path)))
		if '-22.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.monticola',os.path.basename(path)))
		if '-23.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.adansonii',os.path.basename(path)))
		if '-24.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.unicolor',os.path.basename(path)))
		if '-25.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.capensis',os.path.basename(path)))
		if '-26.' in path:
			shutil.copyfile(path, os.path.join(dest,'A.m.ruttneri',os.path.basename(path)))	


prepare_unet(2,4)



























