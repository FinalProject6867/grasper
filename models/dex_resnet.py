import numpy as np
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D, Add, BatchNormalization, Concatenate, Dense, Input
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


def BuildModel():
    print"Setting up model..."
    input_depths = Input(shape=(32, 32, 1))
    input_pose = Input(shape=(7,))
    
    x = Conv2D(16, (3,3), padding='same')(input_depths)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x1 = Conv2D(8, (7,7), padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(32, (3,3), padding='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x2 = Conv2D(32, (5,5), padding='same')(x)
    x = Add()([x1, x2])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (1,1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    x = Concatenate()([x, input_pose])
    x = Dense(2)(x)
    y = Activation('softmax')(x)
    
    model = Model(inputs=[input_depths, input_pose], outputs=y)
    return model
    

