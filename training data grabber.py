# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:59:25 2020

@author: keega
"""
import os
import numpy as np
import librosa

tempdict = dict()
dataDir = "C:/Users/keega/Desktop/Flute"

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
        notesdict[note].append(np.array(tempdict[note]["1.dat"][i]))
        notesdict[note].append(np.array(tempdict[note]["2.dat"][i]))


testchord = (.5*notesdict['A'][0]) + (.5*notesdict['D'][0])

chords = dict()
chords['C Major'] = list()
chords['D Diminished'] = list()
chords ['Eb Minor'] = list()
chords['E Augmented'] = list()

for i in range(len(notesdict['C'])):
    for j in range(len(notesdict['E'])):
        for k in range(len(notesdict['G'])):
            chords['C Major'].append((notesdict['C'][i]+notesdict['E'][j]+notesdict['G'][k])/3)
for i in range(len(notesdict['D'])):     
    for j in range(len(notesdict['Gb'])):
        for k in range(len(notesdict['A'])):
            chords['D Diminished'].append((notesdict['D'][i]+notesdict['F'][j]+notesdict['Ab'][k])/3)
for i in range(len(notesdict['Eb'])):     
    for j in range(len(notesdict['Gb'])):
        for k in range(len(notesdict['Bb'])):
            chords['Eb Minor'].append((notesdict['Eb'][i]+notesdict['Gb'][j]+notesdict['Bb'][k])/3)
for i in range(len(notesdict['D'])):     
    for j in range(len(notesdict['Gb'])):
        for k in range(len(notesdict['A'])):
            chords['E Augmented'].append((notesdict['E'][i]+notesdict['Ab'][j]+notesdict['C'][k])/3)

#features = dict()

#for chord in chords:
#    "perform feature extraction"
#    print(chord)
#
#for feature in features:
#    "find mean"
    
