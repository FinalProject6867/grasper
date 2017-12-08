import numpy as np
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
#from dex_resnet import BuildModel
#from dexnet_inception import BuildModel
from models.gq_cnn import BuildModel
#from models.andreas_net import BuildModel
from keras.utils import to_categorical



def load_data():
    root_path  = 'data/dexnet_data/archive/milestone_data'
    
    depth_maps = np.load(root_path+'/depth.npy')
    hand_poses = np.load(root_path+'/hand_pose.npy')
    
    labels = np.load(root_path+'/grasp_metric.npy')
    labels[labels >= 0.002] = 1
    labels[labels < 0.002] = 0

    labels = to_categorical(labels, num_classes=2)
    
    n, _, _ = depth_maps.shape
    indexes = np.arange(0, n)
   
    # Normalization of Depth Maps
    max_pixel = np.max(depth_maps)
    min_pixel = np.min(depth_maps)
    scale = 2.0/(max_pixel - min_pixel)
    
    depth_maps = scale*depth_maps - 1.0

    #Normalization of hand pose coordinates
    hand_poses = normalize(hand_poses, axis=0, norm='max')

    # Slicing 10% of data for Testing
    x_traini, x_testi, y_train, y_test = train_test_split(indexes, labels, test_size=0.10)
    x_train_maps =  np.expand_dims(depth_maps[x_traini, :, :], axis=-1)
    x_test_maps = np.expand_dims(depth_maps[x_testi, :, :], axis=-1)
    x_train_poses = np.squeeze(hand_poses[x_traini, :4])
    x_test_poses = np.squeeze(hand_poses[x_testi, :4])

    print "Depth Training samples", x_train_maps.shape
    print "Pose Training samples", x_train_poses.shape
    print "Depth Test samples", x_test_maps.shape
    print "Pose Test samples", x_test_poses.shape 
    print "Number of classes in training", np.sum(y_train, axis=0)
    print "Number of classes in testing", np.sum(y_test, axis=0)
    return x_train_maps, x_train_poses, y_train, x_test_maps, x_test_poses, y_test
    

def load_data_batches():
    root_path  = 'data/dexnet_data/balanced_classify_10000_set'
    depth_maps = []
    hand_poses = []
    labels = []
    for i in range(3):
        depth_mapsx = np.load(root_path+str(i)+'/depth.npy')
        hand_posesx = np.load(root_path+str(i)+'/hand_pose.npy')
    
        labelsx = np.load(root_path+str(i)+'/grasp_metric.npy')
        labelsx[labelsx >= 0.002] = 1
        labelsx[labelsx < 0.002] = 0

        labelsx = to_categorical(labelsx, num_classes=2)

        depth_maps.append(depth_mapsx)
        hand_poses.append(hand_posesx)
        labels.append(labelsx)

    depth_maps = np.vstack(depth_maps)
    hand_poses = np.vstack(hand_poses)
    labels = np.vstack(labels)
    n, _, _ = depth_maps.shape
    indexes = np.arange(0, n)
   
    # Normalization of Depth Maps
    max_pixel = np.max(depth_maps)
    min_pixel = np.min(depth_maps)
    scale = 2.0/(max_pixel - min_pixel)
    
    #depth_maps = scale*depth_maps - 1.0

    #Normalization of hand pose coordinates
    #hand_poses = normalize(hand_poses, axis=0, norm='max')

    # Slicing 10% of data for Testing
    x_traini, x_testi, y_train, y_test = train_test_split(indexes, labels, test_size=0.10)
    x_train_maps =  np.expand_dims(depth_maps[x_traini, :, :], axis=-1)
    x_test_maps = np.expand_dims(depth_maps[x_testi, :, :], axis=-1)
    x_train_poses = np.squeeze(hand_poses[x_traini, 1])
    x_test_poses = np.squeeze(hand_poses[x_testi, 1])

    print "Depth Training samples", x_train_maps.shape
    print "Pose Training samples", x_train_poses.shape
    print "Depth Test samples", x_test_maps.shape
    print "Pose Test samples", x_test_poses.shape 
    print "Number of classes in training", np.sum(y_train, axis=0)
    print "Number of classes in testing", np.sum(y_test, axis=0)
    return x_train_maps, x_train_poses, y_train, x_test_maps, x_test_poses, y_test



if __name__ == '__main__':
    model = BuildModel()
    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    tb_call = TensorBoard(log_dir='./balanced_logs', histogram_freq=3)
    print "Loading data..."
    x_train_maps, x_train_poses, y_train, x_test_maps, x_test_poses, y_test = load_data_batches()
    model.fit([x_train_maps, x_train_poses],
              y_train,
              epochs=50,
              verbose=2,
              callbacks=[tb_call],
              validation_split=0.1)

    results = model.evaluate([x_test_maps, x_test_poses],
                             y_test)
    print "TEST => Loss: %0.2f \t Acc: %0.2f" %(results[0], results[1])
    print "Saving..."
    model.save('balanced_depth_resnet.h5')
