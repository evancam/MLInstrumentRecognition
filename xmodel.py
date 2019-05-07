#!/usr/bin/env python
#
#   Created by Evan Cameron
#

from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.xception import Xception

from keras.preprocessing.image import ImageDataGenerator


def get_model(allInstruments):
    
    # Xception as Base model
    xceptionBase = Xception(weights=None, include_top=False, input_shape=(299, 299, 3))
    output = Flatten()(xceptionBase.output)

    # Densely Connected Layer
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)

    # Classification Layer
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(len(allInstruments), activation='softmax')(output)

    model = Model(xceptionBase.input, output)

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model
    