#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 16:57:18 2018

@author: snoopyknight
"""
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

import tensorflow as tf
tf.reset_default_graph()

TRAIN_DIR = '/home/snoopyknight/project/dog_vs_cat/train'
TEST_DIR = '/home/snoopyknight/project/dog_vs_cat/test1'
IMG_SIZE = 50
LR = 1e-3     #LR for learning rate 0.001

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') # just so we remember which saved model is which, sizes must match




def label_img(img):
    word_label = img.split('.')[-3]   #dog.93.png => dog
    # conversion to one-hot array [cat,dog]
    if word_label == 'cat':
        return [1,0]       #[much cat, no dog]
	elif word_label == 'dog':
        return [0,1]	   #[no cat, very doggo]
	else:
        pass
    
    

def create_train_data():
	training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):   #list all file in directory
        label = label_img(img)   #[cat,dog]
		path = os.path.join(TRAIN_DIR,img)
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img) , np.array(label)])
	shuffle(training_data)	#shuffle data
	np.save('train_data.npy',training_data)
	return training_data



def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(TEST_DIR)):   #list all file in directory
		path = os.path.join(TEST_DIR,img)
		img_num = img.split('.')[0]  #for kaggle test data
		img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img) , np.array(img_num)])
	print(testing_data)
	np.save('test_data.npy',testing_data)
	return testing_data



def main():
	train_data = create_train_data()
	# if you already have train data:
	# train_data = np.load('train.npy')


	# 2 layered convolutional neural network, 
	# with a fully connected layer, and then the output layer.
	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 128, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 2, activation='softmax')        #2 class : dog, cat
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')


	#saving our model after every session, and reloading it if we have a saved version
	if os.path.exists('{}.meta'.format(MODEL_NAME)):
		model.load(MODEL_NAME)
		print('model loaded!', MODEL_NAME)
	else: 
		print('nooooo')


	#the training data and testing data are both labeled datasets.
	# The training data is what we'll fit the neural network with
	# the test data is what we're going to use to validate the results.
	train = train_data[:-500]   
	test = train_data[-500:]

	# X:feaure set,  Y:label set
	X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE,IMG_SIZE,1)
	Y = [i[1] for i in train]


	test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE,IMG_SIZE,1)
	test_y = [i[1] for i in test]


	model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
	model.save(MODEL_NAME)

	#if you don't have this file yet
	test_data = process_test_data()
	#if you already have it
	#testing_data = np.load('test_data.npy')

	fig = plt.figure()

	for num, data in enumerate(testing_data[:12]):
		img_num = data[1]
		img_data = data[0]

		y = fig.add_subplot(3,4,index+1)

		orig = img_data
		data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)

		model_out = model.predict([data])[0]
    	#cat:[1,0]
    	#dog:[0,1]
        
        if np.argmax(model_out)==1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'
        
    
        
        
        
        
        
        
        
if __name__ == '__main__':
	main()


# tensorboard --logdir=/home/snoopyknight/project/dog_vs_cat/log
        