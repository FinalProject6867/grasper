import numpy as np
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from dex_resnet import BuildModel


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
    model = BuildModel()
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
