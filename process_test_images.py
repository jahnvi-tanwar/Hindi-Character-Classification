import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py

folder_path = os.getcwd()
input_folder = os.path.join(folder_path,'test','*')
files = glob.glob(input_folder)

features = []
global_feature = []


for file in files:
    path1 = os.path.join(input_folder,file)
    img = plt.imread(path1)
     #img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    global_feature = np.hstack([img])
    features.append(global_feature)

    print('Processed folder {}'.format(file))

#n_samples, nx, ny = np.array(features).shape
#d2_global_features = np.array(features).reshape((n_samples, nx * ny))

h5f_data = h5py.File(os.path.join(folder_path,'testing_data.h5'),'w')
h5f_data.create_dataset('dataset_1',data=np.array(features))
h5f_data.close()