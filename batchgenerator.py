#!/usr/bin/env python
#
#   Created by Evan Cameron
#

import os
import numpy as np
import random
from keras.utils import to_categorical

allInstruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

# Generate a single pair of data and one hot encoded label
def pair_generator(partition, allPaths):
    while True:
        for instrument, npzPaths in allPaths[partition].items():
            filepath = random.choice(npzPaths)
            data = np.load(filepath)['pixels']
            index = allInstruments.index(instrument)
            oneHotLabel = to_categorical(index, len(allInstruments))
            yield data, oneHotLabel


# Generates batches of data
def batch_generator(partition, allPaths, batchSize):
    while True:
        dataPair = pair_generator(partition, allPaths)
        dataBatch, labelBatch = zip(*[next(dataPair) for _ in range(batchSize)])
        dataBatch = np.array(dataBatch)
        labelBatch = np.array(labelBatch)
        yield dataBatch, labelBatch