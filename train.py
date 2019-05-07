#!/usr/bin/env python
#
#   Created by Evan Cameron
#

import glob
import keras
import os
from collections import defaultdict
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.xception import Xception

import xmodel 
from batchgenerator import batch_generator

# Path of the pickle data
dataPath = 'D:/test/square/'

# Path for the trained model output
modelPath = 'D:/model/'

# Instruments in data set
allInstruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

allFilePaths = {
    'train': defaultdict(list),
    'validation': defaultdict(list)
}

# Create a list of .npz file paths for the generators
allNpzFiles = list(glob.glob(os.path.join(dataPath, '**/*.npz')))
for filePath in allNpzFiles:
    instrument = os.path.basename(os.path.dirname(filePath))
    if hash(filePath) % 10 < 8:
        partition = 'train'
    else:
        partition = 'validation'

    allFilePaths[partition][instrument].append(filePath)

# Save the weights of each epoch
weights = keras.callbacks.ModelCheckpoint(
        os.path.join(modelPath, 'weights.{epoch:02d}.h5'),
        verbose=3,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1
    )

model = xmodel.get_model(allInstruments)

model.fit_generator(
        batch_generator('train', allFilePaths, batchSize=12),
        steps_per_epoch=500,
        epochs=30,
        validation_data=batch_generator('validation', allFilePaths, batchSize=12),
        validation_steps=100,
        callbacks=[weights]
)