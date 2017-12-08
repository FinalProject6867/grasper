import numpy as np
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D, Add, BatchNormalization, Concatenate, Dense, Input, MaxPooling2D, LeakyReLU, Lambda, Flatten
from keras import regularizers
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import keras.backend as K


def Tile(x):
    shape = K.shape(x)
    x = K.tile(x, [1, 14*14])
    x = K.reshape(x, (shape[0], 14, 14, 20))
    return x


def BuildModel():

    l = 14*14
    input_depths = Input(shape=(32, 32, 1))
    input_pose = Input(shape=(1,))

    x1 = Conv2D(20, (5,5), padding='valid')(input_depths)
    x1 = LeakyReLU()(x1)
    x1 = MaxPooling2D((2,2))(x1)

    x2 = Dense(20, kernel_regularizer=regularizers.l1(0.1))(input_pose)
    x2 = LeakyReLU()(x2)
    x2 = Lambda(Tile)(x2)

    x3 = Add()([x1, x2])
    x3 = Conv2D(50, (5,5))(x3)
    x3 = LeakyReLU()(x3)
    x3 = MaxPooling2D((2,2))(x3)
    x3 = Flatten()(x3)
    x3 = Dense(20, kernel_regularizer=regularizers.l1(0.1))(x3)
    x3 = LeakyReLU()(x3)
    x3 = Dense(2, kernel_regularizer=regularizers.l1(0.1))(x3)
    y = Activation('softmax')(x3)

    model = Model(inputs=[input_depths, input_pose], outputs=y)
    return model
