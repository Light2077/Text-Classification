import tensorflow.keras.backend as K

from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

import tensorflow as tf
import numpy as np

def accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def precision_recall(y_true, y_pred):

    y_pred = tf.round(tf.keras.backend.clip(y_pred, 0, 1))  # 如果是概率形式的预测标签
    true_positives = tf.reduce_sum(y_true*y_pred)  # 统计所有标签的tp
    predict_positives = tf.reduce_sum(y_pred)  # 统计所有标签的预测是pos的个数
    real_positives = tf.reduce_sum(y_true)  # 统计所有标签实际是pos的个数
    
    # 加一个很小的数防止除0
    precision = true_positives / (predict_positives + 1e-7)  # 预测的对了
    recall = true_positives / (real_positives + 1e-7)
    return precision, recall


def micro_f1(y_true, y_pred):
    """
    y_true : tf.Tensor
    y_pred : tf.Tensor
    """
    y_pred = tf.round(tf.keras.backend.clip(y_pred, 0, 1))  # 如果是概率形式的预测标签
    true_positives = tf.reduce_sum(y_true*y_pred, axis=0)  # 统计所有标签的tp
    predict_positives = tf.reduce_sum(y_pred, axis=0)  # 统计所有标签的预测是pos的个数
    real_positives = tf.reduce_sum(y_true, axis=0)  # 统计所有标签实际是pos的个数
    
    # 加一个很小的数防止除0
    # micro precision and recall
    precision = tf.reduce_sum(true_positives) / (tf.reduce_sum(predict_positives)  + 1e-7)
    recall = tf.reduce_sum(true_positives) / (tf.reduce_sum(real_positives)  + 1e-7)
    micro_f1 = 2 * precision * recall / (precision + recall)

    return micro_f1


def macro_f1(y_true, y_pred):
    """
    y_true : tf.Tensor
    y_pred : tf.Tensor
    """
    y_pred = tf.round(tf.keras.backend.clip(y_pred, 0, 1))  # 如果是概率形式的预测标签
    true_positives = tf.reduce_sum(y_true*y_pred, axis=0)  # 统计每个标签的tp
    predict_positives = tf.reduce_sum(y_pred, axis=0)  # 统计每个标签的预测是pos的个数
    real_positives = tf.reduce_sum(y_true, axis=0)  # 统计每个标签实际是pos的个数
    
    precision = true_positives / (predict_positives + 1e-7)  # 预测的对了
    recall = true_positives / (real_positives + 1e-7)  # 被逮到了的

    macro_f1 = tf.reduce_mean(2 * precision * recall / (precision + recall + 1e-7))

    return macro_f1


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


def evaluation(y_true, y_pred, metrics):
    """
    多标签分类的评估
    metrics : ['accuracy', 'micro_precision', 'macro_precision', 
                'micro_recall', 'macro_recall', 'micro_f1', 'macro_f1']
    """

    y_pred = tf.round(tf.keras.backend.clip(y_pred, 0, 1))

    true_positives = tf.reduce_sum(y_true*y_pred, axis=0)  # 统计每个标签的tp
    predict_positives = tf.reduce_sum(y_pred, axis=0)  # 统计每个标签的预测是pos的个数
    real_positives = tf.reduce_sum(y_true, axis=0)  # 统计每个标签实际是pos的个数

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))  # 准确率无所谓 macro micro

    # micro precision and recall
    micro_precision = tf.reduce_sum(true_positives) / (tf.reduce_sum(predict_positives)  + 1e-9)
    micro_recall = tf.reduce_sum(true_positives) / (tf.reduce_sum(real_positives)  + 1e-9)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    # macro precision and recall
    macro_precision = tf.reduce_mean(true_positives / (predict_positives + 1e-9))
    macro_recall = tf.reduce_mean(true_positives / (real_positives + 1e-9))

    # 不能直接拿平均的p和r来求macro f1
    _precision = true_positives / (predict_positives + 1e-9)
    _recall = true_positives / (real_positives + 1e-9)
    macro_f1 = tf.reduce_mean(2 * _precision * _recall / (_precision + _recall))

    res = dict()
    for metric in metrics:
        if metric == 'accuracy':
            res[metric] = accuracy.numpy()

        if metric == 'micro_precision':
            res[metric] = micro_precision.numpy()

        if metric == 'macro_precision':
            res[metric] = macro_precision.numpy()

        if metric == 'micro_recall':
            res[metric] = micro_recall.numpy()

        if metric == 'macro_recall':
            res[metric] = macro_recall.numpy()

        if metric == 'micro_f1':
            res[metric] = micro_f1.numpy()

        if metric == 'macro_f1':
            res[metric] = macro_f1.numpy()


    return res