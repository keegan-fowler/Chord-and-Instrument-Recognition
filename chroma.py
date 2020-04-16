# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:59:04 2020

@author: keega
"""

import numpy as np
from scipy import fftpack

N = 1024
signal = np.zeros(1024)

def makesignal():
    signal = np.zeros(1024)
    
    for i in range(1024):
        signal[i] = i/2.0
    return signal
def demean(msignal):
    sigmean = np.mean(msignal)
    for i in range(len(msignal)):
        msignal[i] = msignal[i] - sigmean
        
    return msignal

def applyHammingWindow(signal):
    w = np.zeros(1024)
    
    for i in range(1024):
        w[i] = signal[i] * (.54 - .46 * np.cos((2* np.pi * i)/(N)))
    return w


def findFFTMagSqr(w):
    fft = np.fft.fft(w)
    fft = abs(fft)**2
    return fft

def fillPSD(fft):
    PSD = np.zeros(513)
    for i in range(513):
        PSD[i] = fft[i]
    return PSD

def calcPhi():
    fme10 = 2595.0 * np.log10(1.0 + 250.0/700.0)
    fme127 = 2595.0 * np.log10(1.0 + 8000.0/700.0)
    dX = (fme127 - fme10)/27.0
    fme1 = np.zeros(28)
    phi = np.zeros(28)
    for i in range(28):
        fme1[i] = (fme10 + i * dX)
    for i in range(28):
        phi[i] = (((10**(fme1[i]/2595.0))-1.0) * 700.0)
    return fme1,phi

def calcLamda(phi):
    lamba = np.zeros(28)
    for i in range(28):
        lamba[i] = int(phi[i]*513/8000)
    return lamba

def createFilters(lamba):
    filters = np.zeros((26,513))
    for i in range(26):
        for j in range(513):
            if (j < lamba[i]):
                filters[i][j] = 0
            
            elif(j >= lamba[i] and j <= lamba[i+1]):
                filters[i][j] = (j - lamba[i])/(lamba[i+1] - lamba[i])
                
            elif(j > lamba[i+1] and j <= lamba[i+2]):
                filters[i][j] = (lamba[i+2] - j)/(lamba[i+2] - lamba[i+1])
            
            else:
                filters[i][j] = 0
    return filters

def calcEnergy(filters, PSD):
    Y = np.zeros(26)
    for i in range(26):
        esum = 0.0
        for j in range(513):
            esum = esum + filters[i][j] * PSD[j]
        Y[i] = esum
    return Y

def logEnergy(Y):
    X = np.zeros(26)
    for i in range(26):
        X[i] = np.log10(Y[i])
    return X

def genMFCC(X):
    MFCC = np.zeros(13)
    for i in range(13):
        temp = 0
        for j in range(26):
            temp = temp + X[j] * np.cos((i+1)*(j-.5)*np.pi/26.0)
        MFCC[i] = temp
    return MFCC

def getMFCC(signal):
    w = applyHammingWindow(signal)
    fft = findFFTMagSqr(w)
    PSD = fillPSD(fft)
    fme1,phi= calcPhi()
    lamba = calcLamda(phi)
    filters =createFilters(lamba)
    Y = calcEnergy(filters, PSD)
    X = logEnergy(Y)
    MFCC = genMFCC(X)
    return MFCC

def findmean(chromaout):
    chromameans = dict()
    for chord in chromaout:
        chromameans[chord] = np.zeros(12)
        for i in range(12):
            for j in range(1000):
                chromameans[chord][i] += chromaout[chord][j,i]/1000
    return chromameans
            
def findchromatots(PSD):
    chromas = np.zeros(12)
    for k in range(512):
        logvar = np.log2(16000*k/440.0 * 1024)
        roundval = np.round(69.0 + 12 * logvar,0)
        i = roundval % 12
        if i >=0 and i <= 11:
            i = int(i)
            chromas[i] += PSD[k]
    return(chromas)
    
def normalizechromas(chromatot):
    total = 0.
    for j in range(12):
        total += chromatot[j]**2
    total = np.sqrt(total)
    for i in range(12):
        chromatot[i] = chromatot[i]/total
    return chromatot
    
def getchromafeatures(signal):
    newsignal = demean(signal)
    w = applyHammingWindow(newsignal)
    fft = findFFTMagSqr(w)
    PSD = fillPSD(fft)
    chromatot = findchromatots(PSD)
    chroma = normalizechromas(chromatot)
    return chroma

if __name__ == "__main__":
#    signal = ...
#    finalchromas = getchromafeatures(signal)
#    print(finalchromas)
    chromaout = dict()
    for chorditer in chords:
        chromaout[chorditer] = np.zeros((1000,12))
        for index, signal in enumerate(chords[chorditer]):
            chromatemp = getchromafeatures(chords[chorditer][index])
            for index2,point in enumerate(chromatemp):
                chromaout[chorditer][index,index2] = chromatemp[index2]
    chromameans = findmean(chromaout)
    mean4 = chromameans
    
    #pianochroma = getchromafeatures(piano)

totalmean = dict()    
for ins in mean1:
    totalmean[ins] = (mean1[ins]+mean2[ins] + mean3[ins]+mean4[ins])/4
    
#    signal = datadict['raw']
#    w = applyHammingWindow()
#    fft = findFFTMagSqr(w)
#    PSD = fillPSD(fft)
#    fme1,phi= calcPhi()
#    lamba = calcLamda(phi)
#    filters =createFilters(lamba)
#    Y = calcEnergy(filters, PSD)
#    X = logEnergy(Y)
#    MFCC = genMFCC(X)
#    print(MFCC)
#    