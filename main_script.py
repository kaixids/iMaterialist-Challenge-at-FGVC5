# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:29:22 2018

@author: GGmon
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
#from DataGenerator import DataGenerator
from tqdm import tqdm
import keras

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, load_model, Model
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from PIL import ImageFile                            

from sklearn.metrics import hamming_loss

# Parameters
params = {'dim': (224,224),
          'batch_size': 32,
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
input_tensor = Input(shape=(224,224,3))

base_model = keras.applications.mobilenet.MobileNet(input_shape=(224,224,3), include_top=False, weights='imagenet')
last = base_model.output
x = Flatten()(last)
x = Dense(228, activation='softmax')(x)

#Create top model to be fine tuned
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
#top_model.add(Dropout(0.5))
top_model.add(Dense(228, activation='sigmoid'))

len(base_model.layers)

new_model = Sequential()

for layer in base_model.layers[:96]:
    new_model.add(layer)

new_model.add(top_model)

loss_function = hamming_loss()
    
# Compile the model
new_model.compile(loss=hamming_loss, optimizer='nadam', metrics=['accuracy'])

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

new_model.fit_generator(generator = training_generator, 
                        steps_per_epoch = 1000,
                        validation_data = validation_generator,
                        use_multiprocessing=True,
                        workers=6
                        )