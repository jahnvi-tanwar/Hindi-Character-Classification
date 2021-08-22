import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
from sklearn.preprocessing import LabelEncoder

folder_path = os.getcwd()
input_folder = os.path.join(folder_path,'training')
sub_folders = os.listdir(input_folder)

nb_classes = 2
features = []
class_labels = []

for folder in sub_folders:
    path = os.path.join(input_folder,folder,'*')
    files = glob.glob(path)
    current_label = folder

    for file in files:
        path1 = os.path.join(input_folder,folder,file)
        img = plt.imread(path1)
        #img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        global_feature = np.hstack([img])
        class_labels.append(current_label)
        features.append(global_feature)

    print('Processed folder {}'.format(file))

targetNames=np.unique(class_labels)
le=LabelEncoder()
target=le.fit_transform(class_labels)

#n_samples, nx, ny = np.array(features).shape
#d2_global_features = np.array(features).reshape((n_samples, nx * ny))

h5f_data = h5py.File(os.path.join(folder_path,'training_data.h5'),'w')
h5f_data.create_dataset('dataset_1',data=np.array(features))
h5f_label = h5py.File(os.path.join(folder_path,'training_labels.h5'),'w')
h5f_label.create_dataset('dataset_1',data=np.array(target))
h5f_data.close()
h5f_label.close()