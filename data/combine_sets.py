#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy, random, os
import matplotlib.pyplot as plt

def combineSets(set0, set1, newSet):
    folder0 = 'dexnet_data/combine/balanced_classify_5000_set{}/'.format(set0)
    depth0 = numpy.load('{}/depth.npy'.format(folder0))
    grasp0 = numpy.load('{}/hand_pose.npy'.format(folder0))
    metric0 = numpy.load('{}/grasp_metric.npy'.format(folder0))
    pts0 = numpy.load('{}/point_order.npy'.format(folder0))

    folder1 = 'dexnet_data/combine/balanced_classify_5000_set{}/'.format(set1)
    depth1 = numpy.load('{}/depth.npy'.format(folder1))
    grasp1 = numpy.load('{}/hand_pose.npy'.format(folder1))
    metric1 = numpy.load('{}/grasp_metric.npy'.format(folder1))
    pts1 = numpy.load('{}/point_order.npy'.format(folder1))

    with open('{}/ratio.txt'.format(folder0), 'r+') as text0: 
        ratios0 = text0.read()
    with open('{}/ratio.txt'.format(folder1), 'r+') as text1: 
        ratios1 = text1.read()

    all_depth = numpy.concatenate((depth0, depth1))
    all_hand = numpy.concatenate((grasp0, grasp1))
    all_grasp = numpy.concatenate((metric0, metric1))
    all_pts = numpy.concatenate((pts0, pts1))
    ratios = ratios0 + ' . ' + ratios1

    # save everything
    folder_name = 'dexnet_data/balanced_classify_10000_set{}/'.format(newSet)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    numpy.save('{}/depth.npy'.format(folder_name), all_depth)
    numpy.save('{}/hand_pose.npy'.format(folder_name), all_hand)
    numpy.save('{}/grasp_metric.npy'.format(folder_name), all_grasp)
    numpy.save('{}/point_order.npy'.format(folder_name), all_pts)
    with open('{}/ratio.txt'.format(folder_name), 'w') as text_file: 
        text_file.write(ratios)

if __name__ == '__main__':
    #combineSets(0, 1, 3)
    combineSets(2, 3, 4)
    combineSets(4, 5, 5)
    combineSets(6, 7, 6)
    combineSets(8, 9, 7)
