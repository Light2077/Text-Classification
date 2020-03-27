import tensorflow as tf


class TextCNN(tf.keras.Model):
    """
    可以考虑的优化的点：
    - filters 改成可以调的，跟kernel_sizes一样可以变化
    -
    """
    def __init__(self, max_len, vocab_size, embedding_dim, output_dim, 
                 kernel_sizes, filters=2, embedding_matrix=None):
        super(TextCNN, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        if embedding_matrix is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                        weights=[embedding_matrix], trainable=True)
            
        self.kernel_sizes = kernel_sizes  # exm: [2, 3, 4]
        self.filters = filters
        self.conv1d = [tf.keras.layers.Conv1D(filters=self.filters, kernel_size=k, strides=1) 
                       for k in self.kernel_sizes]
        # max_len-k+1 是卷积层后词向量剩下的长度
        # max_len-k+1 这是为了池化层直接取到剩下的词向量的长度
        self.maxpool1d = [tf.keras.layers.MaxPool1D(max_len-k+1) for k in self.kernel_sizes]
        
        self.x_flatten = tf.keras.layers.Flatten()
        # 输出层
        self.dense = tf.keras.layers.Dense(output_dim)
        
    def call(self, x):
        x = self.embedding(x)
        pool_output = []
        for conv, pool in zip(self.conv1d, self.maxpool1d):
            c = conv(x)  # (batch_size, max_len-kernel_size+1, filters)
            p = pool(c)  # (batch_size, 1, filters)
            pool_output.append(p)

        pool_output = tf.concat(pool_output, axis=2)  # (batch_size, 1, n*filters)
        # pool_output = tf.squeeze(pool_output)  # (batch_size, n*filters)
        pool_output = self.x_flatten(pool_output)

        y = self.dense(pool_output)
        return y