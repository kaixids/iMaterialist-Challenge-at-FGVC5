# -*- coding: utf-8 -*-
"""
Created on Sat May 12 15:29:22 2018

@author: GGmon
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from tqdm import tqdm
import keras

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, load_model, Model
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

from PIL import ImageFile                            



from keras import backend as K
from sklearn.metrics import hamming_loss

# Creating loss function

def hamming_loss(y_true, y_pred):
    return K.mean(y_true * (1 - y_pred) + (1 - y_true) * y_pred)
    
# Define a model architecture    

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

for layer in new_model.layers[:96]:
    layer.trainable = False
for layer in new_model.layers[96:]:
    layer.trainable = True

    
# Compile the model
new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', hamming_loss])

x = np.load('train_input_stacked.npy')
y = np.load('train_label_encoded.npy')

history = new_model.fit(x=x, y=y, validation_split=0.2, epochs=1)
