import os
import re
import time
import jieba
import zipfile
import numpy as np
import pandas as pd
import pickle
from functools import wraps
from collections import Counter
from utils.config import root
from multiprocessing import cpu_count, Pool

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.getcwd())
ZIP_DATA = os.path.join(root, 'data', '百度题库.zip')
INPUT_DIR = os.path.join(root,  'data', 'input')
STOP_WORDS = os.path.join(root,  'data', 'stopwords', '哈工大停用词表.txt')
VOCAB = os.path.join(root,  'data', 'vocab.txt')
DATASET = os.path.join(root,  'data', 'dataset.csv')
TOKENIZER_BINARIZER = os.path.join(root, 'data', 'tokenizer_binarizer.pickle')
X = os.path.join(root,  'data', 'x.npy')
Y = os.path.join(root,  'data', 'y.npy')


def count_time(func):
    """
    用于计算函数运行时间的装饰器
    :param func:
    :return:
    """
    @wraps(func)
    def int_time(*args, **kwargs):

        start_time = time.time()  # 程序开始时间
        res = func(*args, **kwargs)
        print('程序{}()共耗时{:.2f}秒'.format(func.__name__, (time.time() - start_time)))
        return res

    return int_time


def unzip_data(file=ZIP_DATA, unzip_dir=INPUT_DIR):
    """
    解压压缩包
    """
    assert os.path.isfile(file)
    if not os.path.isdir(unzip_dir):
        with zipfile.ZipFile(file, 'r') as z:
            z.extractall(unzip_dir)
        print("已将数据解压至{}".format(unzip_dir))
    else:
        print('已存在input文件夹，不进行解压')


def combine_data():
    """
    把四门科目内的所有文件合并
    """
    r = re.compile(r'\[知识点：\]\n(.*)')  # 用来寻找知识点的正则表达式
    r1 = re.compile(r'纠错复制收藏到空间加入选题篮查看答案解析|\n|知识点：|\s|\[题目\]')

    subject_path = os.path.join(INPUT_DIR, '百度题库')
    subjects = os.listdir(subject_path)
    data = []  # 用于存放数据集
    for sub in subjects:
        files = os.listdir(os.path.join(subject_path, sub, 'origin'))  # 获取每门科目下的知识点
        label1 = sub.strip('高中_')  # 主标签
        for f in files:
            label2 = f.strip('.csv')  # 副标签
            df = pd.read_csv(os.path.join(subject_path, sub, 'origin', f))  # 打开csv文件
            df['subject'] = label1  # 主标签：科目
            df['topic'] = label2  # 副标签：科目下主题
            df['category'] = df['item'].apply(lambda x: r.findall(x)[0].replace(',', ' ') if r.findall(x) else '')
            df['item'] = df['item'].apply(lambda x: r1.sub('', r.sub('', x)))
            data.append(df)
    df = pd.concat(data).rename(columns={'item': 'content'}).reset_index(drop=True)
    # df['label'] = df['subject'] + ' ' + df['topic'] + ' ' + df['category']
    df.drop(['web-scraper-order', 'web-scraper-start-url'], axis=1, inplace=True)
    return df


# 有些topic数量本来就小，再这么一分更小了。此函数没有用到
def get_important_label(df, k=5):
    """
    筛选出每个副标签下出现频率最高的k个标签
    return 新数据集
    """

    def get_most_k(df, k):
        l = [i for lb in df.category for i in lb.split()]
        labels = Counter(l)
        most_k = [i[0] for i in labels.most_common()[:k]]
        return most_k

    def label_filter(x, labels):
        """
        x 是df的category列
        """
        x = x.split()
        return ' '.join([i for i in x if i in labels])

    # 找出每个副标签下出现频率最高的k个标签
    tmp = df.groupby(['subject', 'topic']).apply(lambda x: get_most_k(x, k))

    # 把知识点拼接在一起，注意这里面有重复的标签，但是不关键
    labels = [i for lb in tmp for i in lb]  # lb是 主标签-副标签-出现频率最高的k个知识点
    df['label'] = df.category.apply(lambda x: label_filter(x, labels))
    # df = df.drop(df[df.label==''].index, axis=0)
    df['label'] = df['subject'] + ' ' + df['topic'] + ' ' + df['label']
    return df


def label_filter(df, freq=0.01):
    """
    过滤掉出现频率低于一定阈值的知识点标签，科目和主题标签不会过滤
    Parameters
    ----------
    df 总数据集
    freq 出现频率，默认1%

    Returns
    -------
    新数据集 DataFrame格式
    """
    df = combine_data()  # 获得合并的数据集
    categories = ' '.join(df['category']).split()  # 合并
    categories = Counter(categories)
    k = int(df.shape[0] * freq)  # 计算对应频率知识点出现的次数
    most_k = [i[0] for i in categories.most_common() if i[1] > k]  # 过滤掉知识点出现次数小于k的样本

    df.category = df.category.apply(lambda x: ' '.join([label for label in x.split() if label in most_k]))
    df['label'] = df[['subject', 'topic', 'category']].apply(lambda x: ' '.join(x), axis=1)
    return df[['label', 'content']]

def load_dataset():
    return pd.read_csv(DATASET)


def load_stop_words(file=STOP_WORDS):
    stop_words = [line.strip() for line in open(file, encoding='UTF-8').readlines()]
    return stop_words


def sentence_preprocess(sentence, stop_words=None):
    r = re.compile("[^\u4e00-\u9fa5]+|题目")
    sentence = r.sub("", sentence)  # 删除所有非汉字字符
    words = jieba.lcut(sentence, cut_all=False)  # 切词

    # 是否移除停用词，如果传入了停用词参数则移除句子中的停用词
    if stop_words:
        words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@count_time
def df_preprocess(df, stop_words=None):
    df['content'] = df['content'].apply(lambda x: sentence_preprocess(x, stop_words))
    return df


def parallelize(df, func, stop_words=None):
    """
    多核并行处理模块
    有个问题： 多线程处理时，jieba的载入自定义词典失效
    :param df: DataFrame数据
    :return: 处理后的数据
    """
    # func = self.data_frame_proc
    cores = cpu_count() // 2

    print("开始并行处理，核心数{}".format(cores))
    stop_words = [stop_words] * cores
    with Pool(cores) as p:
        df_split = np.array_split(df, cores)  # 数据切分 df->[df1, df2, ... , dfk]

        df = pd.concat(p.starmap(func, zip(df_split, stop_words)))  # 数据分发->处理->合并
    return df


def save_vocab(vocab, path=VOCAB):
    """
    :param path: 要保存的vocab文件路径
    :param vocab: vocab
    """
    with open(path, mode="w", encoding="utf-8") as f:
        for key, value in vocab.items():
            f.write(f'{str(key)} {str(value)}\n')


def load_vocab(file_path=VOCAB, vocab_max_size=None):
    """
    读取字典
    :param file_path: 文件路径 (VOCAB)
    :param vocab_max_size: 最大字典数量
    :return: 返回读取后的字典
    """
    vocab = {}
    reverse_vocab = {}
    for line in open(file_path, "r", encoding='utf-8').readlines():
        word, index = line.strip().split(" ")
        index = int(index)
        # 如果vocab 超过了指定大小
        # 跳出循环 截断
        if vocab_max_size and index > vocab_max_size:
            print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                vocab_max_size, index))
            break
        vocab[word] = index
        reverse_vocab[index] = word
    return vocab, reverse_vocab


def create_tokenizer_binarizer(texts, labels, num_words=50000, tb_save_path=TOKENIZER_BINARIZER):
    text_preprocesser = Tokenizer(num_words=num_words, oov_token="<UNK>")
    text_preprocesser.fit_on_texts(texts)

    # 获得多标签
    mlb = MultiLabelBinarizer()
    mlb.fit(labels.apply(lambda labels: labels.split()))

    tb = {'tokenizer': text_preprocesser, 'binarizer': mlb}
    # 保存

    with open(tb_save_path, 'wb') as f:
        pickle.dump(tb, f)
    print('已创建并保存tokenizer和binarizer至：\n {}'.format(tb_save_path))


def load_tokenizer_binarizer(tb_save_path=TOKENIZER_BINARIZER):
    with open(tb_save_path, 'rb') as f:
        tb = pickle.load(f)
    return tb['tokenizer'], tb['binarizer']


def create_xy(texts, labels, maxlen=150, tb_save_path=TOKENIZER_BINARIZER, x_save_path=X, y_save_path=Y):
    tokenizer, mlb = load_tokenizer_binarizer(tb_save_path)

    # word2index
    x = tokenizer.texts_to_sequences(texts)
    # padding
    x = pad_sequences(x, maxlen=maxlen, padding='post', truncating='post')
    y = mlb.transform(labels.apply(lambda labels: labels.split()))

    np.save(x_save_path, x)
    np.save(y_save_path, y)
    print('已创建并保存x,y至：\n {} \n {}'.format(x_save_path, y_save_path))


def load_xy(x_save_path=X, y_save_path=Y):
    x = np.load(x_save_path)
    y = np.load(y_save_path)
    return x, y


def word2idx(word, vocab, num_words=None):
    try:
        idx = vocab[word]
    except KeyError:
        idx = vocab['<UNK>']
    # 超过最大允许的idx
    if not num_words:
        if idx > num_words-1:
            idx = vocab['<UNK>']
    return idx


def sentence2idxs(sentence, vocab, num_words=None):
    return [word2idx(w, vocab) for w in sentence.split()]


def idx2word(idx, reverse_vocab):
    assert idx <= len(reverse_vocab), '索引超出范围'
    return reverse_vocab[idx]


def idxs2sentence(idxs, reverse_vocab):
    return ' '.join([idx2word(i, reverse_vocab) for i in idxs])


def bert_train_test_split(small=False):
    """
    如果small=True：则是因为自己的电脑太菜，就用比较小的数据量实现模型了
    该函数给bert模型划分了3个数据集
    Returns
    -------

    """
    df = load_dataset()
    df['content'] = df['content'].apply(lambda x:x.replace(' ', ''))
    if small:
        print('use small dataset to test my local bert model really work')
        train = df.sample(128)
        dev = df.sample(64)
        test = df.sample(64)
    else:
        train, test = train_test_split(df, test_size=0.2, random_state=2020)
        train, dev = train_test_split(train, test_size=0.2, random_state=2020)

    print('preprocess for bert!')
    print('create 3 tsv file(train, dev, test)')
    train.to_csv(os.path.join(root,  'data', 'train.tsv'), index=None, sep='\t')
    dev.to_csv(os.path.join(root, 'data', 'dev.tsv'), index=None, sep='\t')
    test.to_csv(os.path.join(root, 'data', 'test.tsv'), index=None, sep='\t')


if __name__ == '__main__':
    unzip_data()  # 解压原始数据
    df = combine_data()  # 合并原始数据
    df = label_filter(df, freq=0.01)  # 过滤掉出现次数小于全部样本数1%的标签

    # 单独为bert划分数据集
    bert_train_test_split()

    # TextCNN数据预处理
    stop_words = load_stop_words()  # 载入停用词
    # df = df_preprocess(df, stop_words)  # 单线程预处理数据
    df = parallelize(df, df_preprocess, stop_words)  # 多线程预处理数据
    print('创建并保存数据集到：', DATASET)
    df[['label', 'content']].to_csv(DATASET, index=None)  # 保存数据集

    texts = df['content']
    labels = df['label']
    create_tokenizer_binarizer(texts, labels, num_words=50000)
    create_xy(texts, labels, maxlen=150)
    
