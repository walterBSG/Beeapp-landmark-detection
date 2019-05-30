#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:01:24 2018

@author: walter
"""

import os
import cv2
import numpy as np
import warnings
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

def weighted_binary_crossentropy(weights):
	
	if isinstance(weights,list) or isinstance(np.ndarray):
		weights=K.variable(weights)
	
	def loss(target, output, from_logits=False):
		
		if not from_logits:
			# transform back to logits
			_epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
			output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
			output = tf.log(output / (1 - output))
	
		return tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=weights)
		
	return loss

def take_imgs(images):
	
	squares = []
	
	for img in images:
		img = img.reshape((400,400,1))
		squares.append(img)

	return np.array(squares)

def predict_with(images, model_path):
	
	warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
	
	X_test = take_imgs(images)
	
	class_weight = [1, 100]
	new_loss = weighted_binary_crossentropy(class_weight)
	
	model = load_model(model_path, custom_objects={'loss': new_loss})
	preds_test = model.predict(X_test, verbose=1)
	
	X_test = np.squeeze(X_test)
	preds_test = np.squeeze(preds_test)
	preds_test_t = (preds_test > 0.8).astype(np.uint8)
	preds_test_t = preds_test_t*255
	return preds_test_t

def load_with(model_path):
	warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
	
	class_weight = [1, 50]
	new_loss = weighted_binary_crossentropy(class_weight)
	
	model = load_model(model_path, custom_objects={'loss': new_loss})
	return model
	
def use_unet(images, model):
	X_test = take_imgs(images)
	
	preds_test = model.predict(X_test, verbose=1)
	
	X_test = np.squeeze(X_test)
	preds_test = np.squeeze(preds_test)
	preds_test_t = (preds_test > 0.8).astype(np.uint8)
	preds_test_t = preds_test_t*255
	return preds_test_t

def take_imgs_full(folder):
    names = []
    images = []
    for filename in os.listdir(folder):
        names.append(filename)
    names = sorted(names, key=str.lower)

    for n in names:
        img = cv2.imread(os.path.join(folder,n), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            names.remove(n)
            
    print(folder," it's OK!!")
    return np.array(images)

def test_unet(folder, model):
	test =   take_imgs_full(folder)
	results = predict_with(test, model)
	
	for index, result in enumerate(results):
		name = '/home/walter/Documents/Results/test/%s.jpg'%index
		cv2.imwrite(name, result)
		name = '/home/walter/Documents/Results/test/blend/%s.jpg'%index
		result = cv2.addWeighted(result,0.6,test[index],0.4,0)
		cv2.imwrite(name, result)

















