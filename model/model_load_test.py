
# coding: utf-8

# In[5]:


# (13300, 128, 129, 1) (5700, 128, 129, 1) (13300, 10) (5700, 10)
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

import os
import keras
import h5py
import librosa
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.models import load_model
#指定显卡
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[ ]:


model = load_model('../models/nbs_vgg16.h5')


# In[ ]:


model.summary()


# In[ ]:


"""
@description: Method to split a song into multiple songs using overlapping windows
"""
def splitsongs(X, y, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)


# In[ ]:


"""
@description: Method to convert a list of songs to a np array of melspectrograms
"""
def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))


# In[ ]:


filename = ['./test/blues.00000.au','./test/blues.00001.au','./test/blues.00002.au','./test/blues.00003.au','./test/blues.00004.au',
            './test/classical.00000.au','./test/classical.00001.au','./test/classical.00002.au','./test/classical.00003.au','./test/classical.00004.au',
            './test/country.00000.au','./test/country.00001.au','./test/country.00002.au','./test/country.00003.au','./test/country.00004.au',
            './test/disco.00000.au', './test/disco.00001.au', './test/disco.00002.au', './test/disco.00003.au', './test/disco.00004.au',
            './test/hiphop.00000.au','./test/hiphop.00001.au','./test/hiphop.00002.au','./test/hiphop.00003.au','./test/hiphop.00004.au',
           './test/jazz.00000.au', './test/jazz.00001.au', './test/jazz.00002.au', './test/jazz.00003.au', './test/jazz.00004.au',
            './test/metal.00000.au','./test/metal.00001.au','./test/metal.00002.au','./test/metal.00003.au','./test/metal.00004.au',
            './test/pop.00000.au', './test/pop.00001.au', './test/pop.00002.au', './test/pop.00003.au', './test/pop.00004.au',
            './test/reggae.00000.au', './test/reggae.00001.au', './test/reggae.00002.au', './test/reggae.00003.au', './test/reggae.00004.au',
            './test/rock.00000.au','./test/rock.00001.au','./test/rock.00002.au','./test/rock.00003.au','./test/rock.00004.au',
           ]
j = 0
temp = [] 
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
for i in filename:
    signal, sr = librosa.load(i)
    signal = signal[:660000]
    window = 0.1
    overlap = 0.5
    xshape = signal.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    # Split the song and create new ones on windows
    temp_X = []
    spsong = [signal[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
    specs = to_melspectrogram(temp_X)
    specs = np.squeeze(np.stack((specs,) * 3, -1))

#     print(specs.shape)
    predictions = model.predict(specs)
#     print(predictions[0].shape)
#     print(np.sum(predictions[0]))
#     print(np.argmax(predictions[0]))
    temp.append(np.argmax(predictions[0]))
    if j<4:
        j = j+1
    else:
        j = 0
        print(temp)
        temp = []

