import os
import csv
import shutil


directory_open = 'clips/'

with open('filtered_files.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for file in reader:
        for filename in os.listdir(directory_open):
            if file[1] == filename:
                directory_save = 'clips_final_net/'
                print(file[1])
                filename = directory_open + file[1]
                directory_save = directory_save + file[1]
                shutil.copyfile(filename, directory_save)
                continue
            else:
                continue