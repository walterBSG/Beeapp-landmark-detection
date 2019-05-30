#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:44:42 2018

@author: walter
"""

import cv2
import os
from utils import takeNamesAndFloats

def makeXML(names, floats, folder):
	cont = 0
	for d, n in enumerate(names):
		img = cv2.imread(n)
		
		rows, cols, channels = img.shape
			
		Bx = int(floats[cont])
		Ux = int(floats[cont])
		By = int(floats[cont+1])
		Uy = int(floats[cont+1])
	
		for x in range (1, 20):
			if (Bx > int(floats[cont])):
				Bx = int(floats[cont])
			if (Ux < int(floats[cont])):
				Ux = int(floats[cont])
			if (By > int(floats[cont+1])):
				By = int(floats[cont+1])
			if (Uy < int(floats[cont+1])):
				Uy = int(floats[cont+1])
			cont += 2
		
		
		if(Ux > cols):
			print(Ux, cols)
		if(Uy > rows):
			print(Uy, rows)
	
		distx = Ux - Bx
		disty = Uy - By
		Bx = Bx - int(distx*0.2)
		if (Bx < 0):
			Bx = 0
	
		By = By - int(disty*0.18)
		if (By < 0):
			By = 0
	
		Ux = Ux + int(distx*0.18)
		if (Ux > cols):
			Ux = cols
	
		Uy = Uy + int(disty*0.2)
		if (Uy > rows):
			Uy = rows
		
		path = os.path.join('/home/walter/Documents/Projeto_Asas/XML', ( os.path.basename(n) + ".xml"))
		f = open(path,"w+")
		f.write("<annotation>\n\t<folder>" + os.path.basename(folder) + "</folder>\n\t<filename>" + os.path.basename(n) + "</filename>\n\t<path>" + n + "</path>\n")
		f.write("\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n")
		f.write("\t<size>\n\t\t<width>" + str(cols) + "</width>\n\t\t<height>" + str(rows) + "</height>\n\t\t<depth>" + str(channels) + "</depth>\n\t</size>\n\t<segmented>0</segmented>\n")
		f.write("\t<object>\n\t\t<name>asa</name>\n\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n")
		f.write("\t\t\t<xmin>" + str(Bx) + "</xmin>\n\t\t\t<ymin>" + str(By) + "</ymin>\n\t\t\t<xmax>" + str(Ux) + "</xmax>\n\t\t\t<ymax>" + str(Uy) + "</ymax>\n\t\t</bndbox>\n\t</object>\n")
	
		f.write("</annotation>")
		f.close

def makeXMLs():
	file = '/home/walter/Documents/ruttner/rutter_correto.txt'  
	folder = '/home/walter/Documents/ruttner/ruttnerCatalogado'
	
	names, floats = takeNamesAndFloats(file, folder)
	makeXML(names, floats, folder)
	
	file = '/home/walter/Documents/Projeto_Asas/acores_correto.txt'  
	folder = '/home/walter/Documents/Projeto_Asas/ASAS_ACORES_2017'
	
	names, floats = takeNamesAndFloats(file, folder)
	makeXML(names, floats, folder)
	
	file = '/home/walter/Documents/Projeto_Asas/peninsula_correto.txt'  
	folder = '/home/walter/Documents/Projeto_Asas/Asas_Peninsula_iberica_FEMEAS'
	
	names, floats = takeNamesAndFloats(file, folder)
	makeXML(names, floats, folder)

makeXMLs()












































