#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy, random
import matplotlib.pyplot as plt

def generateImage(imageName, ptNumber):
    # Pad all image numbers to be five digits
    imagenum = imageName.zfill(5)

    # Read the arrays
    depth_im = numpy.load('{}depth_ims_tf_table_{}.npz'.format(dirpath, imagenum))['arr_0'][ptNumber,...]

    # Reshape both for saving
    depth_edit = depth_im.reshape((32, 32))

    # Create and save image 
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.imshow(depth_edit, cmap='binary')
    ax.axis("off")
    ax.axis("tight")
    plt.savefig('dexnet_data/depth_{}_{}.png'.format(imageName, ptNumber), bbox_inches='tight')
    plt.close()

def detailedImageDataset(fileNum, placeInFile):
    # Pad all image numbers to be five digits
    imagenum = fileNum.zfill(5)

    dirpath = '/media/rachelholladay/Planck/ml_project/3dnet_kit_detection_08_11_17/images/tensors/'

    # Read the arrays

    # both of these are 400 by 400
    depth_im = numpy.load('{}depth_images_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    binary_im = numpy.load('{}binary_images_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    
    # integers
    start_i = numpy.load('{}start_i_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    end_i = numpy.load('{}end_i_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]  

def detailedDataset(fileNum, placeInFile):
    # Pad all image numbers to be five digits
    imagenum = fileNum.zfill(5)

    dirpath = '/media/rachelholladay/Planck/ml_project/3dnet_kit_detection_08_11_17/grasps/tensors/'

    # Read the arrays

    # 32 x 32 depth image
    depth_im = numpy.load('{}depth_ims_tf_table_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    # 7 x 1 vector, hand poses
    hand_config = numpy.load('{}hand_configurations_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    # Values of labels
    robust_grasp_metric = numpy.load('{}robust_ferrari_canny_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    grasp_metric = numpy.load('{}ferrari_canny_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]

    # Intrinsics 4x1, Poses 7x1
    camera_intrs = numpy.load('{}camera_intrs_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    camera_poses = numpy.load('{}camera_poses_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]


    # 4x1 grasp, representing what?
    grasps = numpy.load('{}grasps_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    # 32x32 obj mask
    obj_masks = numpy.load('{}obj_masks_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]

    # Unclear what these represent
    collision_free = numpy.load('{}collision_free_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]   
    image_labels = numpy.load('{}image_labels_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    obj_labels = numpy.load('{}obj_labels_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]

    force_closure = numpy.load('{}force_closure_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    pose_labels = numpy.load('{}pose_labels_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    grasp_labels = numpy.load('{}grasp_labels_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]


def generateDataPoint(pt_num):
    placeInFile = int(str(pt_num)[-3:])
    fileNum = str(pt_num)[:-3]

    # Pad all image numbers to be five digits
    imagenum = fileNum.zfill(5)

    # Read the arrays
    depth_im = numpy.load('{}depth_ims_tf_table_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    hand_pose = numpy.load('{}hand_poses_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]
    grasp_metric = numpy.load('{}robust_ferrari_canny_{}.npz'.format(dirpath, imagenum))['arr_0'][placeInFile,...]

    depth_edit = depth_im.reshape((32, 32))
 
    return (depth_edit, hand_pose, grasp_metric)

def generateRandom(count, threshold):
    # number of files: 6728
    # they each have 1000. but last one has 850
    # so 6727*1000 + 850 data points
    total_pts = 6727*1000 + 850
    ptsList = range(total_pts)
    random.shuffle(ptsList)

    all_depth = numpy.zeros((count, 32, 32))
    all_hand = numpy.zeros((count, 7))
    all_grasp = numpy.zeros((count, 1))
    all_pts = numpy.zeros((count, 1))

    positiveCount = 0
    negativeCount = 0
    desiredSplit = count / 2
    addPt = False
    i = 0
    j = 0
    
    while j < count:
        addPt = False
        print 'Processing {}th data point ..'.format(i),
        depth, hand, grasp = generateDataPoint(ptsList[i])
        if grasp > threshold:
            positiveCount += 1
            if positiveCount <= desiredSplit:
                addPt = True
        else:
            negativeCount += 1
            if negativeCount <= desiredSplit:
                addPt = True

        if addPt:
            print 'adding!'
            all_depth[j] = depth
            all_hand[j] = hand
            all_grasp[j] = grasp
            all_pts[j] = ptsList[i]
            j += 1
        else:
            print 'skipping!'

        i += 1

    print 'Processed {} negatives and {} positives'.format(negativeCount, positiveCount)
    numpy.save('dexnet_data/depth.npy', all_depth)
    numpy.save('dexnet_data/hand_pose.npy', all_hand)
    numpy.save('dexnet_data/grasp_metric.npy', all_grasp)
    numpy.save('dexnet_data/point_order.npy', all_pts)

if __name__ == '__main__':
    #dirpath = '../../3dnet_kit_06_13_17/'
    #thresholdVal = 0.002
    #generateRandom(10000, thresholdVal)
    #generateImage(str(2), 30)

    detailedImageDataset(str(43), 33)
