# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:05:09 2018

@author: Administrator
"""

from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.preprocessing import image
import keras


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.models import Sequential, load_model, Model
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('data/training_images')
valid_files, valid_targets = load_dataset('data/validation_images')
test_files, test_targets = load_dataset('data/testing_images')

def path_to_tensors(image_path):
    
    img = image.load_img(image_path, target_size= (331, 331))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis = 0)

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


#using NASNet pretrained model
model = keras.applications.nasnet.NASNetLarge(input_shape=(331,331,3), include_top=True, weights='imagenet', 
                                              input_tensor=None, pooling=None)


    
def multi_model():    
    model_input = Input(shape=(331, 331, 3))
    x = BatchNormalization()(model_input)
    
    # Define a model architecture
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)       
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
              
    x = GlobalMaxPooling2D()(x)
    
    x = Dense(512, activation='relu')(x)    
    x = Dropout(0.25)(x)
    
    y1 = Dense(228, activation='softmax')(x)
   
    
    model = Model(inputs=model_input, outputs=y1)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

multi_model = multi_model()
