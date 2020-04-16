# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 12:08:39 2020

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
        w[i] = signal[i] * (.54 - .46 * np.cos((2* np.pi * i)/(N-1)))
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
    newsig = demean(signal)
    w = applyHammingWindow(newsig)
    fft = findFFTMagSqr(w)
    PSD = fillPSD(fft)
    fme1,phi= calcPhi()
    lamba = calcLamda(phi)
    filters =createFilters(lamba)
    Y = calcEnergy(filters, PSD)
    X = logEnergy(Y)
    MFCC = genMFCC(X)
    return MFCC

def findmean(MFCCholder):
    mfccmeans = np.zeros(13)
    for note in MFCCholder:
        for i in range(13):
            for j in range(10):
                mfccmeans[i] += MFCCholder[note][j,i]/220
    return mfccmeans
            

if __name__ == "__main__":
    MFCCholder = dict()
    for note in notesdict:
        MFCCholder[note] = np.zeros((10,13))
        for index,signal in enumerate(notesdict[note]):
            mfcctemp = getMFCC(notesdict[note][index])
            for index2,point in enumerate(mfcctemp):
                MFCCholder[note][index,index2] = mfcctemp[index2]
    mfccmeans = findmean(MFCCholder)
    print(mfccmeans)
    
    
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