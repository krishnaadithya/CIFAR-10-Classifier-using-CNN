
# coding: utf-8

# In[11]:


import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from glob import glob
import pickle
from os import listdir
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import time
import tqdm


# In[2]:


from keras.models import Sequential 
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils


# In[22]:


data_path = "E:/ml learning/datasets"
train_data_path = 'E:/ml learning/leaf/train'
#print(os.listdir(train_data_path))
img_length=32
img_breadth=32
num_channels=3
img_size_flat = img_length * img_breadth * num_channels
num_classes=10


# In[4]:


labels=pd.read_csv(os.path.join(data_path,'trainLabels.csv'))
sample_submission=pd.read_csv(os.path.join(data_path,'sampleSubmission.csv'))
print(len(listdir(os.path.join(data_path,'train'))),len(labels))


# In[6]:


_num_files_train=1
_images_per_file=50000
_num_image_train = _num_files_train*_images_per_file


# In[17]:


images=[]
for i in range(1,50001):
    stri=str(i)+'.png'
    image_path=os.path.join(train_data_path,stri)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    imgage_bgt=cv2.resize(image_bgr,(48,48))
    images.append(image_bgr)
x_train = np.array(images)
print(x_train)


# In[8]:


print(x_train)


# In[13]:


le=preprocessing.LabelEncoder()
y_train=le.fit_transform(labels['label'])
y_train=np_utils.to_categorical(y_train)
print(y_train)


# In[157]:


model = Sequential()
model.add(Convolution2D(48, (3, 3), padding='same', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Convolution2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format=None))
model.add(Dropout(0.25))
model.add(Convolution2D(96,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(192, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
start = time.time()
model_info = model.fit(x_train, y_train, batch_size=128, epochs=200, verbose=1)

