#!/usr/bin/env python
#
#   Created by Evan Cameron
#

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from scipy import misc, ndimage
from scipy.signal import windows
from customstft import save_stft

# Root directories for where the audio files are and where to put the images
audioroot = './IRMAS-TrainingData/IRMAS-TrainingData'
imageroot = 'D:/test/'

#All windows (Name, function, overlap % ) that are used  
windows = [ ['square', windows.boxcar, 0], 
            ['hanning', np.hanning, 0.5], 
            ['bartlett', np.bartlett, 0.5], 
            ['kaiser5', np.kaiser, 0.705], 
            ['HFT248D', windows.general_cosine, 0.841]]

for window in windows:

    imageWindowRoot = imageroot+window[0]
    
    # Save images of the log scaled STFTs
    for audioPath in list(glob.glob(audioroot+'/**/*.wav')):

        imagePath=audioPath.replace(audioroot,imageWindowRoot)
        pathlib.Path(os.path.dirname(imagePath)).mkdir(parents=True, exist_ok=True)
        imageBase, _ = os.path.splitext(imagePath) 
        save_stft(audioPath, window[1], 1024, window[2], imageBase, 'npz')