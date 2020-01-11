from os import path
from pydub import AudioSegment
import glob
import csv
import os
# files
dst = "clips_converted/"

directory_open = 'clips_final_net/'
# convert wav to mp3


for filename in os.listdir(directory_open):
    directory_save = 'clips_converted/'
    directory_save = directory_save + filename
    path = directory_open+filename
    sound = AudioSegment.from_mp3(path)
    #sound.export(directory_save, format="wav")