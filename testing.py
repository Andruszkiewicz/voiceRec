from keras.models import load_model
import librosa
import csv
import numpy as np
import glob
from sklearn.preprocessing import LabelBinarizer

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
max_length = 1000
DATA_DIR= "original_files/"
files = glob.glob(DATA_DIR + "*.wav")
detection_model_path = "checkpoints_original/voice_recognition_best_model_32.hdf5"
voice_classifier = load_model(detection_model_path, compile=False)

count = 0
for file_path in files:
    #file_path = "original_files/original_common_voice_tt_17343438.wav"
    waves, sr = librosa.load(file_path, mono=True)
    print("The talking Tatar is : ")
    filename = file_path.split('\\')[1]
    filename = filename.split('.')[0] + ".mp3"
    filename = filename.split('_', 1)[1]
    true_label =''
    with open('filtered_files.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for file in reader:
            if filename == file[1]:
                true_label = file[0]
                print(true_label)
                break
            else:
                continue
    # y = pd.get_dummies(y).as_matrix()
    mfcc = librosa.feature.mfcc(waves, sr)
    mfcc = np.pad(mfcc, ((0, 0), (0, max_length - len(mfcc[0]))), mode='constant', constant_values=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    preds = voice_classifier.predict(mfcc)
    print(preds)
    predsinv = label_binarizer.inverse_transform(preds)
    print("I think that the talking Tatar is : ")
    print(predsinv)
    if (predsinv == true_label):
        print("Correct!")
        count += 1

    else:
        print("Incorrect!")

print("Wynik poprawnych to: ")
print(count)

    #preds = np.argmax(preds, axis=1)
    #print("To jes preds")
    #print(preds)
    #preds = preds[0,:]
    #print(preds)
    #print(preds)
    #print(preds)
    #voice_probability = np.max(preds)
    #print("Amount of preds: ")
    # print(preds.shape)
    # for value,person in zip(preds,labels):
    #     for person_value in value:
    #         print(person_value, person)
    #         if(person_value ==true_label):
    #             print("Correct")
    #         else:
    #             print("Incorrect")
    #print(preds.argmax())
    # print("To jest argument max")
    # print(preds.argmax())
    #label = labels[preds.argmax()]
