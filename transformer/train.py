
import os
import sys
import time
import argparse
import tensorflow as tf
sys.path.append(os.getcwd())

from utils.preprocess import load_testcnn_data
from utils.metrics import micro_f1, macro_f1, precision_recall
from transformer.model import TransformerClassifier

# Masking
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  # 序列中是0的填True然后 bool 转 float

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)  # 维度扩展两维，why?


def create_look_ahead_mask(size):
    # num_lower 表示保留多少条下对角线，-1表示全部保留
    # num_upper 表示保留多少条上对角线，0表示全部不保留
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), num_lower=-1, num_upper=0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# 自定义学习率衰减
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)  # 取开方的倒数
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def predict(model, x, batch_size=1024):
    dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
    res = []
    for batch_x in dataset:
        enc_padding_mask = create_padding_mask(batch_x)
        
        y_pred = model(batch_x, training=False, enc_padding_mask=enc_padding_mask)
        res.append(y_pred)
        
    res = tf.concat(res, axis=0)
    return res




def evaluation(model, x, y):
    y = tf.cast(y, dtype=tf.float32)
    y_pred = predict(model, x)
    
    predict_accuracy = tf.keras.metrics.BinaryAccuracy(name='predict_accuracy')
    acc = predict_accuracy(y, y_pred)
    mi_f1 = micro_f1(y, y_pred)
    ma_f1 = macro_f1(y, y_pred)
    precision, recall = precision_recall(y, y_pred)
    
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


if __name__ == '__main__':

	# 数据
	train_x, dev_x, test_x, train_y, dev_y, test_y = load_testcnn_data()
	train_dataset = load_dataset(train_x, train_y)
	args = get_params()

	# 参数
	num_layers = args.num_layers
	d_model = args.d_model
	num_heads = args.num_heads
	dff = args.dff
	vocab_size = 50000
	maximum_position_encoding = args.maximum_position_encoding
	output_dim = len(train_y[0])  # 97
	dropout_rate = args.dropout_rate
	steps_per_epoch = len(train_x) // BATCH_SIZE
	epochs = args.epochs

	# 模型
	model = TransformerClassifier(num_layers, d_model, num_heads, dff, vocab_size, 
                                    maximum_position_encoding, output_dim, dropout_rate)

	# 学习率
	learning_rate = CustomSchedule(d_model)

	# 配置优化器
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
	                                     epsilon=1e-9)
	# 评估指标
	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

	# 损失函数
	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='auto')

	# checkpoint
	checkpoint_path = "notebook/Transformer/checkpoints/train"

	ckpt = tf.train.Checkpoint(transformer=model,
	                           optimizer=optimizer)

	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

	# 如果检查点存在，则恢复最新的检查点。
	if ckpt_manager.latest_checkpoint:
	    ckpt.restore(ckpt_manager.latest_checkpoint)
	    print ('Latest checkpoint restored!!')

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

	    enc_padding_mask = create_padding_mask(x)
	    with tf.GradientTape() as tape:
	        y_pred = model(x, training=True, enc_padding_mask=enc_padding_mask)
	        loss = loss_object(y, y_pred)

	    grads = tape.gradient(loss, model.trainable_variables)
	    optimizer.apply_gradients(zip(grads, model.trainable_variables))

	    train_loss(loss)
	    train_accuracy(y, y_pred)

	    mi_f1=micro_f1(y, y_pred)
	    ma_f1=macro_f1(y, y_pred)
	    return mi_f1, ma_f1, y_pred

	# 训练
	for epoch in range(epochs):
	    start = time.time()

	    train_loss.reset_states()
	    train_accuracy.reset_states()

	    for batch, (x, y) in enumerate(train_dataset.take(steps_per_epoch)):
	        mi_f1 ,ma_f1, _ = train_step(x, y)

	        if batch % 20 == 0:
	            print('epoch {} batch {:3d} loss {:.4f} acc {:.4f} micro_f1 {:.4f} macro_f1 {:.4f}'.format(
	                epoch+1, batch+1, train_loss.result(), 
	                train_accuracy.result(), mi_f1, ma_f1))

	    # 每隔5轮保存一下        
	    if (epoch + 1) % 5 == 0:
	        ckpt_save_path = ckpt_manager.save()
	        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
	                                                         ckpt_save_path))

	    print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
	    evaluation(model, dev_x, dev_y)
	    print('Time taken for 1 epoch {:.2f} sec\n'.format(time.time() - start))

