import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from textcnn.model import TextCNN
from utils.preprocess import load_testcnn_data

parser = argparse.ArgumentParser(__doc__)

parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--embedding_dim", type=int, default=256)
parser.add_argument("--filters", type=int, default=2)
parser.add_argument("--kernel_sizes", type=str, default='2,3,4')
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--dropout_rate", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--checkpoint_path", type=str, default="../notebook/TextCNN/checkpoints/train")

args = parser.parse_args()


def predict(model, x, batch_size=1024):
    dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    res = []
    for batch_x in dataset:
        y_pred = model(batch_x)
        res.append(y_pred)
    res = tf.concat(res, axis=0)
    return res


def load_dataset(x, y):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(256, reshuffle_each_iteration=True)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    # 数据
    train_x, dev_x, test_x, train_y, dev_y, test_y = load_testcnn_data()
    train_dataset = load_dataset(train_x, train_y)

    # 参数
    vocab_size = 50000
    output_dim = len(train_y[0])  # 97
    steps_per_epoch = len(train_x) // args.batch_size
    kernel_sizes = [int(i) for i in args.kernel_sizes.split(',')]
    # kernel_sizes = [2, 3, 4]
    # 模型
    model = TextCNN(args.max_len, vocab_size, args.embedding_dim,
                    output_dim, kernel_sizes, args.filters)

    # 配置优化器
    optimizer = tf.keras.optimizers.Adam(args.learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    # 评估指标
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # 损失函数
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='auto')

    # checkpoint
    ckpt = tf.train.Checkpoint(textcnn=model,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint_path, max_to_keep=1)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # 训练
    # 该 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地
    # 执行。该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变
    # 批次大小（最后一批次较小）导致的再追踪，使用 input_signature 指定
    # 更多的通用形状。
    # 这里填的128指的是句子的长度
    train_step_signature = [
        tf.TensorSpec(shape=(None, 128), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(x, y):

        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_object(y, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss(loss)
        return y_pred

    # 训练
    for epoch in range(args.epochs):
        start = time.time()
        train_loss.reset_states()

        for batch, (x, y) in enumerate(train_dataset.take(steps_per_epoch)):
            y_pred = train_step(x, y)

            if batch % 20 == 0:
                print('epoch {} batch {:3d} loss {:.4f}'.
                      format(epoch+1, batch+1, train_loss.result()))

        # 每隔5轮保存一下
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.
                  format(epoch+1, ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.
              format(epoch + 1, train_loss.result()))

        print('Time taken for 1 epoch {:.2f} sec\n'.
              format(time.time() - start))
