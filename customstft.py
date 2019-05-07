#!/usr/bin/env python
#
#   Created by Evan Cameron
#
#   Derived from work by Frank Zalkow, 2012-2013 (CC BY 3.0)

import math
import numpy as np
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
from PIL import Image
from scipy.signal import windows
from scipy import misc, ndimage

# Short time fourier transform of audio signal
def stft(sig, frameSize, overlapFac, window):

    win = set_window(frameSize, window)

    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(math.floor(frameSize/2.0)), sig)    
    # The number of frames used in the windowing function
    numFrames = math.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
        
    frames = stride_tricks.as_strided(samples, shape=(numFrames, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)

def set_window(frameSize, window):
    
    
    # Pick windowing function 
    if (window == windows.general_cosine):
        # Coefficients for the HFT248D general cosine window            
        HFT248D = [1, 1.985844164102, 1.791176438506, 1.282075284005,
                    0.667777530266, 0.240160796576, 0.056656381764, 0.008134974479,
                    0.000624544650, 0.000019808998, 0.000000132974]    

        win = window(frameSize, HFT248D, sym=False)

    elif (window == np.kaiser):
        win = window(frameSize, 5)

    else:
        win = window(frameSize)
    
    return win


# Creates and saves a plotted stft
def save_stft(audiopath, window, binsize, overlap, plotpath=None, fileFormat='jpg'):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize, overlap, window)

    # Amplitude to decibel
    ims = 20.*np.log10(np.abs(s)/10e-6)
    
    timebins, freqbins = np.shape(ims)
    
    fig = plt.figure(figsize=(15, 7.5))
    fig.tight_layout()
    r = fig.canvas.get_renderer()
    spectogram = plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap="jet", interpolation="none")
    plt.axis('off')
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    spectogram.axes.get_xaxis().set_visible(False)
    spectogram.axes.get_yaxis().set_visible(False)


    if fileFormat == 'npz':
        fig.draw(r)
        # Convert from plot to rgb pixel array
        nCols, nRows = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_rgb()
        data = np.fromstring(buf, dtype=np.uint8).reshape(nRows, nCols, 3)
        resizedImage = np.array(Image.fromarray(data).resize((299, 299)))

        np.savez(plotpath+'.npz', pixels=resizedImage)

    else:
        plt.savefig(plotpath+'.'+fileFormat, bbox_inches="tight", pad_inches = 0)
        
    plt.clf()
    plt.close('all')