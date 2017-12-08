import numpy as np
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D, Add, BatchNormalization, Concatenate, Dense, Input, MaxPooling2D, LeakyReLU, Lambda, Flatten
from keras import regularizers
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import keras.backend as K



def BuildModel():
    
    input_depths = Input(shape=(32, 32, 1))
    input_pose = Input(shape=(1,))

    x1 = Conv2D(64, (7,7), padding='valid')(input_depths)
    x1 = Activation('relu')(x1)

    x1 = Conv2D(64, (5,5), padding='valid')(x1)
    x1 = Activation('relu')(x1)

    x1 = MaxPooling2D((2,2))(x1)
    x1 = Conv2D(64, (3,3), padding='valid')(x1)
    x1 = Activation('relu')(x1)

    x1 = Conv2D(64, (3,3), padding='valid')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024)(x1)
    x1 = Activation('relu')(x1)

    x2 = Dense(16)(input_pose)
    x2 = Activation('relu')(x2)
    
    x3 = Concatenate()([x1, x2])
    x3 = Dense(1024)(x3)
    x3 = Activation('relu')(x3)
    out = Dense(2)(x3)
    out = Activation('softmax')(out)


    model = Model(inputs=[input_depths, input_pose], outputs=out)
    return model
