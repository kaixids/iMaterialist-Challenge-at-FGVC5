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

data = json.load(open('train.json'))

columns = ['imgID', 'url', 'labelID']



with open('train.json', 'r') as file:    
    obj = json.load(file)
    train_set = pd.DataFrame(obj['images'], columns=['url', 'imageID'])
    annotations = pd.DataFrame(obj['annotations'], columns = ['labelID', 'imageID'])
    

train_set.head()

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
    


labels = []
labels_val = []
for i in range(len(train)):
    labels += train['labelId'][i]
    
for i in range(len(validation)):
    labels_val += train['labelId'][i]
    
label_counter = collections.Counter(labels)
print(label_counter.most_common())

#come up with a list of labels to search for in data
search_list = []
for i in range(len(label_counter.most_common())):
    if label_counter.most_common()[i][1] < 1000:
        search_list.append(label_counter.most_common()[i][0])

print(len(search_list)

#iterate through the entire training dataset and search for imageId's that have those labels:
image_Ids = []
for i in range(len(train)):
     if any(x in train['labelId'][i] for x in search_list):
         image_Ids.append(train['imageId'][i])

search_list3 = []
for i in range(len(label_counter.most_common())):
    if label_counter.most_common()[i][1] < 10000:
        search_list3.append(label_counter.most_common()[i][0])
print(len(search_list3))      
       
image_Ids3 = []
for i in range(len(train)):
     if any(x in search_list3 for x in train['labelId'][i]):
         image_Ids3.append(train['imageId'][i])

new_train = train[train['imageId'].isin(image_Ids3)] 


new_train.head()
new_labels = new_train['labelId'].tolist()
new_labels = [i for sublist in new_labels for i in sublist]
        
new_counter = collections.Counter(new_labels).most_common()
print(label_counter.most_common())
print(new_counter)

new_search_list = []
for i in range(len(new_counter)):
    if new_counter[i][1] < 1000:
        new_search_list.append(new_counter[i][0])
        
print(len(new_search_list))
print(len(search_list))

#-------------------next iteration-------------------
new_search_list2 = []
for i in range(len(new_counter)):
    if new_counter[i][1] < 5000:
        new_search_list2.append(new_counter[i][0])
print(len(new_search_list2))      
       
image_Ids4 = []
for i in range(len(new_train)):
     if any(x in new_search_list2 for x in new_train['labelId'][i]):
         image_Ids4.append(new_train['imageId'][i])

new_train2 = new_train[new_train['imageId'].isin(image_Ids3)] 


new_train2.head()
new_labels2 = new_train2['labelId'].tolist()
new_labels2 = [i for sublist in new_labels2 for i in sublist]
        
new_counter = collections.Counter(new_labels).most_common()
print(label_counter.most_common())
print(new_counter)

new_search_list = []
for i in range(len(new_counter)):
    if new_counter[i][1] < 1000:
        new_search_list.append(new_counter[i][0])
        
print(len(new_search_list))
print(len(search_list))



for i 
labels  =set(labels)

test.head()

len(labels)
img_dict = dict.fromkeys(labels)
img_dict

max_num_img = 500
max_img_ID = ''

label_list = []

for i in range(len(train)):
    label_list += train['labelId'][i]
    counter = collections.Counter(label_list)
    if counter.most_common()[-30][1] > max_num_img:
        max_img_ID = train['imageId'][i]
        break
    

counter = collections.Counter(label_list)
print(counter.most_common()[-1][1])

print(type(counter.most_common()[-30][1]))

def download(data_type, data):

    #create a directory if it is absent
    directory = 'data/'+data_type+'_images/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(len( data)):
        #response = requests.get(data.loc[i]['url'], timeout = 5, stream = True)

        try:
            response = requests.get(data.loc[i]['url'], timeout = 50000, stream = True)

            with open( directory+ str(data.loc[i]['imageId'])+ '.jpeg', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
        except ( requests.exceptions.ConnectionError, urllib3.exceptions.ProtocolError ):
            print("Connection refused")

    return 1


def convertToParallelSolution (ind, data_type, data, numOfProcesses):
    totalnumberOfSamples = len( data )

    numOfBins = round ( totalnumberOfSamples / numOfProcesses ) + 1
    start =  int (ind * numOfBins )
    end = int (start + numOfBins )

    result = 0
    
    if end >= totalnumberOfSamples:
        end = totalnumberOfSamples

    if end <= start:
        return result 

    if end > start:
        result = download (data_type, data[start : end].reset_index( ) )

    print("Batch {} of {} is done so far!!!!! {}.json (in progress)".format (ind, numOfProcesses, data_type))
    #return result
    return 1


def parallel_solution (data_type, data, numOfProcesses=20 ):

    pool = Pool(processes=numOfProcesses)              # start 20 worker processes

    # launching multiple evaluations asynchronously *may* use more processes
    multiple_results = [pool.apply_async(convertToParallelSolution, args=(ind, data_type, data, numOfProcesses, ))for ind in range(numOfProcesses)]

    resultContainer =  [res.get() for res in multiple_results]


download()

if __name__ == "__main__":

    dataTypeLst = ["test", "train", "validation" ]
    for data_type in dataTypeLst:
        data = json.load(open(data_type + '.json'))
        data = json_normalize(data["images"])
        print("{}.json is fully loaded".format (data_type))
        parallel_solution  (data_type, data )



