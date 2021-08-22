import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import json
# from utils.io import write_json

folder_path = os.getcwd()

h5f_data = h5py.File(os.path.join(folder_path,'training_data.h5'),'r')
h5f_label = h5py.File(os.path.join(folder_path,'training_labels.h5'),'r')
global_features_string_train = h5f_data['dataset_1']
global_labels_string_train = h5f_label['dataset_1']

global_features_train = np.array(global_features_string_train)
global_labels_train = np.array(global_labels_string_train)

h5f_data.close()
h5f_label.close()

h5f_data = h5py.File(os.path.join(folder_path,'testing_data.h5'),'r')
global_features_string_test = h5f_data['dataset_1']

global_features_test = np.array(global_features_string_test)

h5f_data.close()

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(global_features_train, global_labels_train, epochs=10)

test_predict = cnn.predict(global_features_test)

i = 0

all_images =  os.listdir(os.path.join(os.getcwd(),'test'))

res = {}

for img in all_images:
    if(np.argmax(test_predict[i]) == 1):
        res[img] = 1
    else:
        res[img] = 0
    i+=1

def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)

def generate_sample_file(filename):
    write_json(filename, res)



generate_sample_file('./result.json')
