# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:08:37 2020

@author: keega
"""
import numpy as np
mfccmeans = dict()
chrommeans = dict()

from scipy import fftpack

N = 1024


mfccmeans['trombone'] = np.array([45.0942,14.4574,17.4655,1.23223,14.6037,3.30578,15.1347,3.55915,13.9366,3.48739,12.3773,2.85096,10.7808])
mfccmeans['trumpet'] = np.array([38.7932,-5.94111, 14.4597,-1.09646,17.2875,3.434,16.2059,2.46789,13.8876,1.38848,12.3827,0.986482,11.7776])
mfccmeans['flute'] = np.array([28.5399,-2.36262,15.6747,-1.04278,17.5827,0.647211,12.7232,0.638618,12.753,2.37457,12.5154,2.68266,10.6592])
mfccmeans['saxophone'] = np.array([31.7621,-2.86236,20.8573,4.83253,15.4331,0.36758,16.3982,1.90975,14.4076,0.293002,11.3965,1.42097,11.4903])

chrommeans['C Major'] = np.array([0.367651,0.0227917,0.230376,0.0795502,0.26779,0.0374084,0.0887473,0.560353,0.294718,0.0133881,0.0157946,0.168713])
chrommeans['D Diminished'] = np.array([0.225903,0.0907167,0.185545,0.122605,0.0128223,0.701561,0.0872099,0.00253581,0.33358,0.113054,0.0322999,0.00726112])
chrommeans['E Augmented'] = np.array([0.472808,0.0344098,0.0184992,0.186871,0.381402,0.0590366,0.0391697,0.0707016,0.468787,0.105205,0.0309785,0.239506])
chrommeans['Eb Minor'] = np.array([0.00242437,0.253625,0.135875,0.307513,0.0190417,0.234668,0.5505,0.0678189,0.0292478,0.0447426,0.523928,0.0488255])


def demean(msignal):
    sigmean = np.mean(msignal)
    for i in range(len(msignal)):
        msignal[i] = msignal[i] - sigmean
        
    return msignal

def findclosest(sigmfcc):
    mfccdiff = dict()
    for instrument in mfccmeans:
        mfccdiff[instrument] = mfccmeans[instrument] - sigmfcc
    mfccdist = dict()
    for instrument in mfccdiff:
        total = 0
        for value in mfccdiff[instrument]:
            total += value**2
        mfccdist[instrument] = np.sqrt(total)
    lowestval = 99999999
    for instrument in mfccdist:
        if mfccdist[instrument] <= lowestval:
            lowestval = mfccdist[instrument]
            result = instrument
    return result
    
def testsignal(item):
    sigmfcc = getMFCC(item)
    result = findclosest(sigmfcc)
    return result
    
def findclosestchrom(sigchrom):
    chromdiff = dict()
    for chord in chrommeans:
        chromdiff[chord] = chrommeans[chord] - sigchrom
    chromdist = dict()
    for chord in chromdiff:
        total = 0
        for value in chromdiff[chord]:
            total += value**2
        chromdist[chord] = np.sqrt(total)
    lowestval = 99999999
    for chord in chromdist:
        if chromdist[chord] <= lowestval:
            lowestval = chromdist[chord]
            result = chord
    return result

def testchromsignal(item):
    sigchrom = getchromafeatures(item)
    result = findclosestchrom(sigchrom)
    return result

def makesignal():
    signal = np.zeros(1024)
    
    for i in range(1024):
        signal[i] = i/2.0
    return signal

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
    chordres = dict()
    for chord in chords:
        chordres[chord] = list()
        for item in chords[chord]:
            q = item
            chordres[chord].append(testchromsignal(item))
    correct = 0
    total = 0
    
    for chord in chordres:
        print(chord)
        correct = 0.0
        total = 0.0
        for sig in chordres[chord]:
            if sig == chord:
                correct += 1
            total += 1
        print(correct/total)
    percentage = correct/total 
    print(percentage)
#    
    
    
    
#   ahmfcc = getMFCC(ah)
#   closest = findclosest(ahmfcc)
 