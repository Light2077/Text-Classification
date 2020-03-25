import os
import sys
import time
import argparse
import tensorflow as tf
from utils.metrics import micro_f1, macro_f1
sys.path.append(os.getcwd())
from utils.preprocess import load_testcnn_data
from textcnn.model import TextCNN

def predict(model, x, batch_size=1024):
    dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    res = []
    for batch_x in dataset:
        y_pred = model(batch_x)
        res.append(y_pred)
        
    res = tf.concat(res, axis=0)
    return res


def evaluation(model, x, y):
    y = tf.cast(y, dtype=tf.float32)
    y_pred = predict(model, x)
    
    predict_accuracy = tf.keras.metrics.BinaryAccuracy(name='predict_accuracy')
    acc = predict_accuracy(y, y_pred)
    mi_f1=micro_f1(y, y_pred)
    ma_f1=macro_f1(y, y_pred)
    
    print("val accuracy {:.4f}, micro f1 {:.4f} macro f1 {:.4f}".format(
                    acc.numpy(), mi_f1.numpy(), ma_f1.numpy()))
    return acc, mi_f1, ma_f1
	

def load_dataset(x, y):
	dataset = tf.data.Dataset.from_tensor_slices((x, y))
	dataset = dataset.cache()
	dataset = dataset.shuffle(BUFFER_SIZE,reshuffle_each_iteration=True).batch(BATCH_SIZE, drop_remainder=True)
	dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
	return dataset

def get_params():
	parser = argparse.ArgumentParser()
	parser.add_argument("--num_layers", default=4, help="transformer encoder layer nums", type=int)
	parser.add_argument("--d_model", default=128, help="word vector dims", type=int)
	parser.add_argument("--num_heads", default=8, help="num heads", type=int)
	parser.add_argument("--dff", default=512, help="feed forward neural netword hidden units", type=int)
	parser.add_argument("--maximum_position_encoding", default=10000, type=int)
	parser.add_argument("--output_dim", default=97, type=int)
	parser.add_argument("--dropout_rate", default=0.1, type=float)
	parser.add_argument("--epochs", default=5, type=int)
	args = parser.parse_args()
	return args

BUFFER_SIZE = 256
BATCH_SIZE = 128