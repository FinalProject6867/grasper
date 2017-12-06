#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy, random
import matplotlib.pyplot as plt

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
        # Do randomized repeated sampling
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
    #TODO more specific file names
    numpy.save('dexnet_data/depth.npy', all_depth)
    numpy.save('dexnet_data/hand_pose.npy', all_hand)
    numpy.save('dexnet_data/grasp_metric.npy', all_grasp)
    numpy.save('dexnet_data/point_order.npy', all_pts)

if __name__ == '__main__':
    #TODO correct path
    dirpath = '../../3dnet_kit_06_13_17/'
    thresholdVal = 0.002
    generateRandom(10000, thresholdVal)
