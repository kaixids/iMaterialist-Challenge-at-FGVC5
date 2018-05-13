# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:29:22 2018

@author: GGmon
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from DataGenerator import DataGenerator
from tqdm import tqdm
import keras

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, load_model, Model
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

from PIL import ImageFile                            
# Parameters
params = {'dim': (331,331),
          'batch_size': 31,
          'n_classes': 228,
          'n_channels': 3,
          'shuffle': True}

# Datasets
df = pd.read_csv('reduced_train.csv')
train_path = 'data/training_images/'
validation_path = 'data/validation_images/'

partition = {}

labels = {}

train_ID = []
for i in tqdm(range(len(df))):
    #gather list of IDs   
    img_path = train_path + str(df['imageId'][i]) +'.jpeg'
    if os.path.isfile(img_path):
        train_ID.append(str(df['imageId'][i]))
        labels[str(df['imageId'][i])] = df['labelId'][i]
        #train_data.append(x)
        
val_ID = []
for i in tqdm(range(len(validation))):
    img_path = validation_path + str(validation['imageId'][i]) +'.jpeg'
    if os.path.isfile(img_path):
        val_ID.append('v'+str(validation['imageId']))
        labels[('v'+str(df['imageId'][i]))] = validation['labelId'][i]

partition['train'] = train_ID
partition['validation'] = val_ID

# Define a model architecture    
base_model = keras.applications.nasnet.NASNetLarge(input_shape=(331,331,3), include_top=True, weights='imagenet', 
                                              input_tensor=None, pooling=None)    

model = Sequential()
model.add(Dense(228, input_shape=base_model.output_shape[1:], activation='softmax'))
     

    
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

model.fit_generator(generator = training_generator,
                    steps_per_epoch = 1000,
                    validation_data = validation_generator,
                    use_multiprocessing=True,
                    workers=6
                    )