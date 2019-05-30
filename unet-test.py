#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:59:55 2019

@author: walter
"""
import os
import cv2
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import backend as K
from skimage.io import imshow
import BlobDetection
import procrustes_module as pr
from utils import takeFilesByExtension, shift_angle


class_weight = [1, 50]

def weighted_binary_crossentropy(weights):
	if isinstance(weights,list) or isinstance(np.ndarray):
		weights= K.variable(weights)
	
	def loss(target, output, from_logits=False):
		if not from_logits:
			# transform back to logits
			_epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
			output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
			output = tf.log(output / (1 - output))
	
		return tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=weights)
		
	return loss

new_loss = weighted_binary_crossentropy(class_weight)

print('Taking data')

def take_imgs(folder, clahe=False, threshold=False):
    names = []
    images = []
	
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	
    for filename in os.listdir(folder):
        names.append(filename)
    names = sorted(names, key=str.lower)
	
    for n in names:
        img = cv2.imread(os.path.join(folder,n), cv2.IMREAD_GRAYSCALE)
		
        if clahe:
            img = clahe.apply(img)
        if threshold:
            _, img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY);
			
        if img is not None:
            img = img.reshape((400,400,1))
            images.append(img)
        else:
            names.remove(n)
            
    print(folder," it's OK!!")
    return np.array(images)

modelName = '/home/walter/Documents/cv2/raio-4-plateau-dust-pesos-kernel5.h5'
model = load_model(modelName, custom_objects={'loss': new_loss})

X_test =    take_imgs('/home/walter/Documents/Results/wings', clahe=True)
y_test =    take_imgs('/home/walter/Documents/Projeto_Asas/test_masks', threshold=True)/255

preds_test = model.predict(X_test, verbose=1)

X_test = np.squeeze(X_test)
y_test = np.squeeze(y_test)
preds_test = np.squeeze(preds_test)

# Threshold predictions
#fazer um print de antes e depois
preds_test_t = (preds_test > 0.93).astype(np.uint8)

ix = random.randint(0, len(preds_test_t))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(y_test[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()

preds_test_t = preds_test_t*255
for ind, img in enumerate(preds_test_t):
    name = "/home/walter/Documents/Results/test/%s.jpg"%ind
    cv2.imwrite(name, img)
    blend = cv2.addWeighted(img,0.6,X_test[ind],0.4,0)
    name = "/home/walter/Documents/Results/test/blend/%s.jpg"%ind
    cv2.imwrite(name, blend)

preds_test_t = shift_angle(preds_test_t)

y_test = y_test.astype(np.uint8)*255
y_test = shift_angle(y_test)
# detect the center of the "dots" created by the unet
dots_list = BlobDetection.take_points(preds_test_t)
print(len(y_test))

dots_list = pr.list_dot_check(dots_list)

result = BlobDetection.compare(preds_test_t, y_test)

print(len(dots_list))
print(len(result))































