import cv2
import numpy

def regrade3(old, new, point):
	a = int(round(((new*point)/old),0))
	return a

def reshape(img):

	desired_size = 400 #novo tamanho de imagem
	
	img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2GRAY)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	print(img.shape)
	rows, cols = img.shape


	old_size = img.shape[:2] # old_size is in (height, width) format

	ratio = float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])

	# new_size should be in (width, height) format
	img = cv2.resize(img, (new_size[1], new_size[0]))

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [0, 0, 0]
	new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	
	return new_im