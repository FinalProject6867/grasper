#/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(norm_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes) #, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = norm_cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        reg = format(cm[i, j], 'd')
        norm = format(norm_cm[i, j], '.2f')
        txt = '{} ({})'.format(reg, norm)
        plt.text(j, i, txt, 
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
#np.set_printoptions(precision=2)

cnf_matrix = np.array([[0, 17706], [0, 152664]])
class_names = ['Success', 'Failure']

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='')
plt.savefig("unbalanced_paper.pdf")

plt.show()
