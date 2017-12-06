#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy, random, os
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

def generateRandom(count, threshold, num_set):
    # number of files: 6728
    # they each have 1000. but last one has 850
    # so 6727*1000 + 850 data points
    total_pts = 6727*1000 + 850
    ptsList = range(total_pts)

    all_depth = numpy.zeros((count, 32, 32))
    all_hand = numpy.zeros((count, 7))
    all_grasp = numpy.zeros((count, 1))
    all_pts = numpy.zeros((count, 1))

    positiveCount = 0
    negativeCount = 0
    desiredSplit = count / 2
    addPt = False
    j = 0
    
    while j < count:
        addPt = False
        print 'Trying for {}th data point ..'.format(j),
        # Repeat random sampling..
        idx = numpy.random.choice(ptsList, 1)[0]
        depth, hand, grasp = generateDataPoint(idx)
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
            all_pts[j] = idx
            j += 1
        else:
            print 'skipping!'

    folder_name = 'dexnet_data/balanced_classify_{}_set{}/'.format(count, num_set)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    numpy.save('{}/depth.npy'.format(folder_name), all_depth)
    numpy.save('{}/hand_pose.npy'.format(folder_name), all_hand)
    numpy.save('{}/grasp_metric.npy'.format(folder_name), all_grasp)
    numpy.save('{}/point_order.npy'.format(folder_name), all_pts)
    with open('{}/ratio.txt'.format(folder_name), 'w') as text_file:
        ratios = 'Processed {} negatives and {} positives'.format(negativeCount, positiveCount) 
        text_file.write(ratios)

if __name__ == '__main__':
    dirpath = '/media/rachelholladay/Planck/ml_project/3dnet_kit_06_13_17/'
    thresholdVal = 0.002
    numPts = 10000
    for i in xrange(3):
        generateRandom(numPts, thresholdVal, i)
