# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:19:04 2020

@author: keega
"""
import os
import numpy as np
import librosa

tempdict = dict()
dataDir = "C:/Users/keega/Desktop/Trumpet"

for directory in os.listdir(dataDir):
    tempdict[directory] = dict()
    for file in os.listdir(dataDir + '/' + directory):
        tempdict[directory][file] = dict()
        print(file)
        f = open(dataDir + '/' + directory +'/' + file,'r')
        firstline = f.readline()
        for i in range(5):
            tempdict[directory][file][i] = []
            for j in range(1024):
                tempdict[directory][file][i].append(int(f.readline()[:-3]))

notesdict = dict()
for note in tempdict:
    notesdict[note] = list()
    for i in range(5):
        notesdict[note].append(np.array(tempdict[note]["1.txt"][i]))
        notesdict[note].append(np.array(tempdict[note]["2.txt"][i]))


testchord = (.5*notesdict['A'][0]) + (.5*notesdict['D'][0])

chords = dict()
chords['C Major'] = list()
chords['D Major'] = list()

for i in range(len(notesdict['C'])):
    for j in range(len(notesdict['E'])):
        for k in range(len(notesdict['G'])):
            chords['C Major'].append((notesdict['C'][i]+notesdict['E'][j]+notesdict['G'][k])/3)
for i in range(len(notesdict['D'])):     
    for j in range(len(notesdict['Gb'])):
        for k in range(len(notesdict['A'])):
            chords['D Major'].append((notesdict['D'][i]+notesdict['Gb'][j]+notesdict['A'][k])/3)

features = dict()
for chord in chords:
    "perform feature extraction"
    print(chord)

for feature in features:
    "find mean"
    
