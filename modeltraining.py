import pickle
import numpy as np
import pandas as pd
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from featureextraction import extract_features
import csv
import sklearn
#from speakerfeatures import extract_features
import warnings
import glob
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelBinarizer
import glob
import numpy as np
import random
import librosa
import wave
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# path to training data
# source   = "development_set/"
DATA_DIR ="clips_downsampled/"
SEED = 2017
files = glob.glob(DATA_DIR + "*.wav")
X_train, X_val = train_test_split(files, test_size=0.2, random_state=SEED)


print('# Training examples: {}'.format(len(X_train)))
print('# Validation examples: {}'.format(len(X_val)))

labels = []
with open('filtered_files.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for file in reader:
        label = file[0]
        if label not in labels:
            labels.append(label)
print(labels)

label_binarizer = LabelBinarizer()
label_binarizer.fit(list(set(labels)))

def one_hot_encode(x): return label_binarizer.transform(x)

n_features = 20
max_length = 1000
n_classes = len(labels)

def batch_generator(data, batch_size=16):
    while 1:
        random.shuffle(data)
        X, y = [], []
        for i in range(batch_size):
            wav = data[i]
            waves, sr = librosa.load(wav, mono=True)
            filename = wav.split('\\')[1]
            filename = filename.split('.')[0] + ".mp3"
            filename = filename.split('_', 1)[1]
            with open('filtered_files.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                for file in reader:
                    if filename == file[1]:
                        label = file[0]
                        y.append(label)
                        break
                    else:
                        continue


            #y = pd.get_dummies(y).as_matrix()
            mfcc = librosa.feature.mfcc(waves, sr)
            mfcc = np.pad(mfcc, ((0,0), (0, max_length - len(mfcc[0]))), mode='constant', constant_values=0)
            X.append(np.array(mfcc))
        yP = np.asarray(y)
        # yP = pd.get_dummies(yP).as_matrix()
        yFinal = one_hot_encode(yP)
        yield np.array(X), np.array(yFinal)

learning_rate = 0.001
batch_size = 64
n_epochs = 50
dropout = 0.5

input_shape = (n_features, max_length)
steps_per_epoch = 50
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=input_shape,
               dropout=dropout))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(n_classes, activation='softmax'))

opt = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt,
metrics=['accuracy'])
model.summary()

callbacks = [ModelCheckpoint('checkpoints/voice_recognition_best_model_{epoch:02d}.hdf5', save_best_only=True),
            EarlyStopping(monitor='val_acc', patience=2)]

#batch_generator(X_train, batch_size)

history = model.fit_generator(
    generator=batch_generator(X_train, batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    verbose=1,
    validation_data=batch_generator(X_val, 32),
    validation_steps=5,
    callbacks=callbacks
)