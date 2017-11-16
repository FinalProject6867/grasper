import numpy as np
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D, Add, BatchNormalization, Concatenate, Dense, Input
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

def load_data():
    depth_maps = np.load('../data/depth.npy')
    hand_poses = np.load('../data/hand_pose.npy')
    n, _, _ = depth_maps.shape
    indexes = np.arange(0, n)
    
    labels = np.load('../data/grasp_metric.npy')
    labels[labels >= 0.002] = 1
    labels[labels < 0.002] = 0

    labels = to_categorical(labels, num_classes=2)
    x_traini, x_testi, y_train, y_test = train_test_split(indexes, labels, test_size=0.10)
    x_train_maps = depth_maps[x_traini, :, :]
    x_test_maps = depth_maps[x_testi, :, :]
    x_train_poses = hand_poses[x_traini, :]
    x_test_poses = hand_poses[x_testi, :]
    
    return x_train_maps, x_train_poses, y_train, x_test_maps, x_test_poses, y_test


if __name__ == '__main__':
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
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    tb_call = TensorBoard(log_dir='./logs', histogram_freq=3)
    print "Loading data..."
    x_train_maps, x_train_poses, y_train, x_test_maps, x_test_poses, y_test = load_data()

    model.fit([x_train_maps, x_train_poses],
              y_train,
              epochs=50,
              verbose=2,
              callbacks=[tb_call],
              validation_split=0.1,
              class_weights={0:0.20, 1:0.80})

    results = model.evaluate([x_test_maps, x_test_poses],
                             y_test)
    print "TEST => Loss: %0.2f \t Acc: %0.2f" %(results[0], results[1])
    print "Saving..."
    model.save('depth_resnet.h5')

    

