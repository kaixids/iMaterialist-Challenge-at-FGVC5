 # -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:30:58 2018

@author: Administrator
"""

import json
from pprint import pprint
import urllib
import pandas as pd
import requests
import numpy as np
import time
from sklearn.preprocessing import MultiLabelBinarizer
import os
import multiprocessing
from multiprocessing import Pool
import shutil
from pandas.io.json import json_normalize
import urllib3
import collections
from tqdm import tqdm



def loading_json():
    script_start_time = time.time()
    
    print('%0.2f min: Start loading data'%((time.time() - script_start_time)/60))
    
    train={}
    test={}
    validation={}
    with open('train.json') as json_data:
        train= json.load(json_data)
    with open('test.json') as json_data:
        test= json.load(json_data)
    with open('validation.json') as json_data:
        validation = json.load(json_data)
    
    print('Train No. of images: %d'%(len(train['images'])))
    print('Test No. of images: %d'%(len(test['images'])))
    print('Validation No. of images: %d'%(len(validation['images'])))
    
    # JSON TO PANDAS DATAFRAME
    # train data
    train_img_url=train['images']
    train_img_url=pd.DataFrame(train_img_url)
    train_ann=train['annotations']
    train_ann=pd.DataFrame(train_ann)
    train=pd.merge(train_img_url, train_ann, on='imageId', how='inner')
    
    # test data
    test=pd.DataFrame(test['images'])
    
    # Validation Data
    val_img_url=validation['images']
    val_img_url=pd.DataFrame(val_img_url)
    val_ann=validation['annotations']
    val_ann=pd.DataFrame(val_ann)
    validation=pd.merge(val_img_url, val_ann, on='imageId', how='inner')
    
    datas = {'Train': train, 'Test': test, 'Validation': validation}
    for data in datas.values():
        data['imageId'] = data['imageId'].astype(np.uint32)
    
    print('%0.2f min: Finish loading data'%((time.time() - script_start_time)/60))
    print('='*50)
    
    print('%0.2f min: Start converting label'%((time.time() - script_start_time)/60))
    
    mlb = MultiLabelBinarizer()
    train_label = mlb.fit_transform(train['labelId'])
    validation_label = mlb.transform(validation['labelId'])
    dummy_label_col = list(mlb.classes_)
    print(dummy_label_col)
    print('%0.2f min: Finish converting label'%((time.time() - script_start_time)/60))
    
    for data in [validation_label, train_label, test]:
        print(data.shape)
    
    return train, test, validation
        

def shrink_df(df, threshold = 5000):
    
    #count the labels in the given dataframe
    labels = []
    for i in range(len(df)):
        labels += df['labelId'][i]
        
    label_counter = collections.Counter(labels).most_common()
    print(label_counter)
    
    #come up with a list of labels to search for in data
    search_list = []
    for i in range(len(label_counter)):
        # if the count for specific label is below threshold, then actively search for it
        if label_counter[i][1] < threshold:
            search_list.append(label_counter[i][0])
    
    print(len(search_list))
    
    #iterate through the entire training dataset and search for imageId's that have those labels:
    image_Ids = []
    for i in range(len(df)):
         if any(x in search_list for x in df['labelId'][i]):
             image_Ids.append(df['imageId'][i])
    
    # filtering for the new dataframe
    new_df = df[df['imageId'].isin(image_Ids)] 
    #reset the index
    new_df = new_df.reset_index(drop = True)
        
    return new_df


train, test, validation = loading_json()

train.index
train.columns
df = shrink_df(train, 8000)


#run shrink_df a few times to get the dataframe we want
df2 = shrink_df(df, 5000)
df2 = shrink_df(df2, 1000)

final_df = shrink_df(df2, 500)

del train, df, df2

def download(data_type, data):

    #create a directory if it is absent
    directory = 'data/'+data_type+'_images/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(7875,len(data)):
        #response = requests.get(data.loc[i]['url'], timeout = 5, stream = True)

        try:
            response = requests.get(data.loc[i]['url'], timeout = 50000, stream = True)

            with open( directory+ str(data.loc[i]['imageId'])+ '.jpeg', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
        except ( requests.exceptions.ConnectionError, urllib3.exceptions.ProtocolError ):
            print("Connection refused")

    return 1

download('training', df2)
download('validation', validation)

download('testing', test)


# *********************************************************************
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
import ast
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True            

     
train_path = 'data/training_images/'
train_batch = os.listdir(train_path)

final_df.index
final_df['imageId'][1]


# Load the filtered dataframe
df = pd.read_csv('reduced_train.csv')

mlb = MultiLabelBinarizer()
train_label = mlb.fit_transform(df['labelId'])

mlb.classes_

df['labelId']
ast.literal_eval(df.labelId[0])

train_data = []

for i in tqdm(range(len(df))):    
    try:
        img_path = train_path + str(df['imageId'][i]) +'.jpeg'
        img = image.load_img(img_path, target_size= (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)        
        train_data.append(x)
        
    except:
        print('no file', final_df['imageId'][i])

train_labels = []

for i in tqdm(range(len(df))):
    img_path = train_path + str(df['imageId'][i]) +'.jpeg'
    if os.path.isfile(img_path):
        x = df['labelId'][i]
        x = ast.literal_eval(x)
        #x = np.expand_dims(x, axis = 0)
        train_labels.append(x)
        

mlb = MultiLabelBinarizer()
train_label = mlb.fit_transform(train_labels)

# Create a list of np array for encoded labels
train_label_encoded = []
for i in range(len(train_label)):
    x = np.expand_dims(i, axis = 0)
    train_label_encoded.append(x)

train_label_encoded = np.vstack(train_label_encoded)

from sklearn.preprocessing import MultiLabelBinarizer
one_hot = MultiLabelBinarizer()
one_hot.fit_transform(train_labels)



train_label = np.vstack(train_labels)

        
validation_path = 'data/validation_images/'

for i in tqdm(range(len(validation))):
    
    try:
        img_path = validation_path + str(validation['imageId'][i]) +'.jpeg'
        img = image.load_img(img_path, target_size= (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        np.save(os.path.join('mobilenet_train_data', ('v' + str(validation['imageId'][i]))), x)
        #train_data.append(x)
        
    except:
        print('no file', validation['imageId'][i])


testing_path = 'data/testing_images/'
test_data = []
null_list = []
for i in tqdm(range(20000,30000)):
    
    try:
        img_path = testing_path + str(test['imageId'][i]) +'.jpeg'
        img = image.load_img(img_path, target_size= (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #np.save(os.path.join('mobilenet_train_data', ('v' + str(validation['imageId'][i]))), x)
        test_data.append(x)
        
    except:
        null_list.append(test['imageId'][i])
        print('no file', test['imageId'][i])
        
        
del testing_stack2
testing_stack3 = np.vstack(test_data)
np.save('testing_stack3', testing_stack3)


np.save('train_input_stacked', train_input)

np.save(img_array, train_input)
        
final_df['img_processed'] = train_data

processed_df = final_df[final_df['img_processed'] != 'Missing_File']




if __name__ == "__main__":

    dataTypeLst = ["test", "train", "validation" ]
    for data_type in dataTypeLst:
        data = json.load(open(data_type + '.json'))
        data = json_normalize(data["images"])
        print("{}.json is fully loaded".format (data_type))
        parallel_solution  (data_type, data )
        
        
        