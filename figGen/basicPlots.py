#/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy, csv
import matplotlib.pyplot as plt
from ss_plotting.make_plots import plot

networks = ['andreas', 'gqcnn', 'inception', 'res_net']
networkName = networks[3]

def readCSV(filename):
    dataStorage = []
    with open('data/recent_results/{}_splits/{}.csv'.format(networkName, filename), 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            pieces = row[0].split(',')
            if pieces[0] == 'Wall':
                continue
            floatPieces = [float(k) for k in pieces]
            dataStorage.append(floatPieces)
    return numpy.array(dataStorage)

def plotAccuracy(train, val):
    plot([[train[:, 1], train[:, 2]],
          [val[:, 1], val[:, 2]]],
          series_colors=['green', 'red'],
          series_labels=['Training', 'Validation'],
          plot_ylim=[0, 0.8],
          show_plot=False,
          plot_xlabel='Number of Steps',
          plot_ylabel='Accuracy',
          #savefile_size=(2.5, 1.5),
          #plot_title='Accuracy using Inception Net',
          savefile='figs/{}_accuracy.pdf'.format(networkName))

def plotLoss(train, val):
    plot([[train[:, 1], train[:, 2]], 
          [val[:, 1], val[:, 2]]], 
          series_colors=['green', 'red'],
          series_labels=['Training', 'Validation'], 
          plot_ylim=[0, 9],
          show_plot=False,
          plot_xlabel='Number of Steps', 
          plot_ylabel='Loss',
          #plot_title='Loss using Inception Net',
          savefile='figs/{}_loss.pdf'.format(networkName))

if __name__ == '__main__':
    train_acc = readCSV('train_acc')
    valid_acc = readCSV('val_acc')
    plotAccuracy(train_acc, valid_acc)
    train_loss = readCSV('train_loss')
    valid_loss = readCSV('val_loss')
    plotLoss(train_loss, valid_loss)
