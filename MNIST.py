#!/usr/bin/env python
# coding: utf-8

# In[69]:


import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, MaxPool2D, Dense
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# In[70]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[71]:


train_images.shape


# In[72]:


len(train_labels)


# In[73]:


train_labels


# In[74]:


test_images.shape


# In[75]:


len(train_labels)


# In[76]:


test_labels


# In[77]:


from tensorflow.keras import models
from tensorflow.keras import layers
net = models.Sequential()
net.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
net.add(layers.Dense(10, activation='softmax'))


# In[78]:


net.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[79]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# In[80]:


from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[88]:


net.fit(train_images, train_labels, epochs=6, batch_size=128)


# In[90]:


test_loss, test_acc = net.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

