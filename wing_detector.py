
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:00:33 2018

@author: walter
"""
import numpy as np
import os
import tensorflow as tf
import resize_square as rs
from utils import list_paths
import cv2
import imp

utils_ops = imp.load_source('ops', '/home/walter/Documents/models/research/object_detection/utils/ops.py')
label_map_util = imp.load_source('label_map_util', '/home/walter/Documents/models/research/object_detection/utils/label_map_util.py')
vis_util = imp.load_source('visualization_utils', '/home/walter/Documents/models/research/object_detection/utils/visualization_utils.py')


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
def prepare():
	# What model to take.
	MODEL_NAME = '/home/walter/Documents/models/research/object_detection/mobilenet'
	
	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
	
	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join('/home/walter/Documents/models/research/object_detection/', 'itens.pbtxt')
	
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
		
	category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
	
	return (detection_graph, category_index)



def run_inference_for_single_image(image, graph, sess):
	# Get handles to input and output tensors
	ops = tf.get_default_graph().get_operations()
	all_tensor_names = {output.name for op in ops for output in op.outputs}
	tensor_dict = {}
	for key in ['num_detections', 'detection_boxes', 'detection_scores',
			'detection_classes', 'detection_masks']:
		tensor_name = key + ':0'
		if tensor_name in all_tensor_names:
			tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
					tensor_name)
	image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

	# Run inference
	output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

	# all outputs are float32 numpy arrays, so convert types as appropriate
	output_dict['num_detections'] = int(output_dict['num_detections'][0])
	output_dict['detection_classes'] = output_dict[
			'detection_classes'][0].astype(np.uint8)
	output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
	output_dict['detection_scores'] = output_dict['detection_scores'][0]
	if 'detection_masks' in output_dict:
		output_dict['detection_masks'] = output_dict['detection_masks'][0]
	
	return output_dict

def detectFromFolder(folder):

	paths = list_paths(folder)
	
	images = detectFromPaths(paths)

	return images

def detectFromFolderWithClass(paths):
	
	
	images = detectFromPaths(paths)

	return images

def detectFromPaths(paths):

	images = []
	names = []
	detection_graph, category_index = prepare()
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	
	with detection_graph.as_default():
		with tf.Session() as sess:
			
			for path in paths:
				img = cv2.imread(path)
				rows, cols, _ = img.shape
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
				output_dict = run_inference_for_single_image(img, detection_graph, sess)
				
				#detectar quantos objetos foram detectados
				boxes = sum(i > 0.5 for i in output_dict['detection_scores'])
				
				for b in range(boxes):
					
					#"desnormalizar" TODO função desnormalizar
					By = output_dict['detection_boxes'][b][0] * rows
					Bx = output_dict['detection_boxes'][b][1] * cols
					Uy = output_dict['detection_boxes'][b][2] * rows
					Ux = output_dict['detection_boxes'][b][3] * cols
					
					new_img = img[int(By):int(Uy),int(Bx):int(Ux)]
					new_img = rs.reshape(new_img)
					new_img = clahe.apply(new_img)
					
					images.append(new_img)
					names.append(os.path.basename(path))
	
	return images, names