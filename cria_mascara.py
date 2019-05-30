import cv2
import os
import numpy as np

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
					line = line.replace('.JPG','.jpg')
				names.append(line.replace('\n',''))
				
	return names, floats

def createMask(names, floats, folder, save_folder, size):
	
	images = []
	
	for name in names:
		images.append(cv2.imread(os.path.join(folder,name),0))
	
	cont = 0
	
	for d, img in enumerate(images):
		
		if d == 4:
			print(names[4])
		rows, cols = img.shape
		i = np.zeros((rows, cols, 1), np.uint8)
	
		for x in range (1, 20):
	
			xs = int(floats[cont])
			ys = int(floats[cont+1])
	
	
			i = cv2.circle(i, (xs, ys), size,255,-1)
			cont += 2
	
		nome = os.path.join(save_folder,names[d])
		cv2.imwrite(nome, i)
	

def create_masks(folder, save_folder, size):
	
	names, floats = takeFloatsAndNames('/home/walter/Documents/new_data/acores_400.txt')
	createMask(names, floats, folder, save_folder, size)
	names, floats = takeFloatsAndNames('/home/walter/Documents/new_data/peninsula_400.txt')
	createMask(names, floats, folder, save_folder, size)
	names, floats = takeFloatsAndNames('/home/walter/Documents/new_data/ruttner_400.txt')
	createMask(names, floats, folder, save_folder, size)
	
	
def test_mask(original_folder, mask_folder, test_folder):
	originals = os.listdir(original_folder)
	
	for original in originals:
		
		original_img = cv2.imread(os.path.join(original_folder, original),0)
		mask = cv2.imread(os.path.join(mask_folder, original),0)
		
		new_img = cv2.addWeighted(mask ,0.6,original_img,0.4,0)
		
		cv2.imwrite(os.path.join(test_folder, original), new_img)

#original_folder = '/home/walter/Documents/Projeto_Asas/test'
#mask_folder = '/home/walter/Documents/Projeto_Asas/test_masks'
#test_folder = '/home/walter/Documents/Results/test'
#test_mask(original_folder, mask_folder, test_folder)