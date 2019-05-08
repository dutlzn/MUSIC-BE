import os
import keras
import h5py
import librosa
import itertools
import numpy as np
import matplotlib.pyplot as plt 
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.models import load_model

from werkzeug.utils import secure_filename

from flask import Flask, flash, request, redirect, url_for
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api
from flask_jsonpify import jsonify

from json import dumps

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# load model, pring summary of the model 
global model 
model = load_model('./model/nbs_vgg16.h5')

global graph
graph = tf.get_default_graph()

global genres
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
# print(model.summary())


UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['mp3','au','wav'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)
# 跨域访问
CORS(app)

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


#----------send info----------------

@app.route("/")
def hello():
    return jsonify({'text':'Hello World!'})                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
     

#test send info
class Genres_Info(Resource):
    def get(self):
        result = {'genres':'bules'}
        return jsonify(result)
api.add_resource(Genres_Info, '/genres_info') # Route_4

#-------------get file----------------------------

def allowed_file(filename):
    #就是允许接受的文件类型 
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/get_file', methods=['POST'])
def upload_file():
    print(request.files)
    # check if the post request has the file part
    if 'file' not in request.files:
        print('no file in request')
        return""
    file = request.files['file']
    with graph.as_default():
        filename = file.filename
        # filename = secure_filename(file.filename) #会过滤中文名字 别加这个比较好 因为加了之后，
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(filename)
        signal, sr = librosa.load(filename)
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
        print(specs.shape)
        predictions = model.predict(specs)
        print(predictions[0].shape)

    if file.filename == '':
        print('no selected file')
        return""

    if file and allowed_file(file.filename):
        # print("接受到了文件")
        # filename = file.filename
        # # filename = secure_filename(file.filename) #会过滤中文名字 别加这个比较好 因为加了之后，
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print(filename)
        # signal, sr = librosa.load(filename)
        # signal = signal[:660000]
        # window = 0.1
        # overlap = 0.5
        # xshape = signal.shape[0]
        # chunk = int(xshape*window)
        # offset = int(chunk*(1.-overlap))
        # # Split the song and create new ones on windows
        # temp_X = []
        # spsong = [signal[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
        # temp_X = [s for s in spsong]
        # # for s in spsong:
        # #     temp_X.append(s)
        # specs = to_melspectrogram(temp_X)
        # specs = np.squeeze(np.stack((specs,) * 3, -1))
        # print(specs.shape)
        # predictions = model.predict(specs)
        # print(np.argmax(predictions[0]))
        return ""
    print("end")
    return 0




if __name__ == '__main__':
   app.run(port=5000)