#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time    : 2020/2/8 7:40 下午 
# @Author  : Roger 
# @Version : V 0.1
# @Email   : 550997728@qq.com
# @File    : metrics.py
import tensorflow.keras.backend as K

from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

import tensorflow as tf
import numpy as np


def f1_np(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return micro_f1, macro_f1


def softmax_cross_entropy(preds, labels):
    """
    Softmax cross-entropy loss with masking.
    """
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def accuracy(preds, labels):
    """
    Accuracy with masking.
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def micro_f1(y_true, y_pred):
    """
    F1 metric.

    Computes the micro_f1 and macro_f1,
    metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """
    Micro_F1 metric.
    """
    precision = K.sum(true_positives) / (K.sum(predicted_positives) + K.epsilon())
    recall = K.sum(true_positives) / (K.sum(possible_positives) + K.epsilon())
    micro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return micro_f1


def macro_f1(y_true, y_pred):
    """
    F1 metric.

    Computes the micro_f1 and macro_f1,
    metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """
    Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    macro_f1 = K.mean(2 * precision * recall / (precision + recall + K.epsilon()))
    return macro_f1
