import os
import re
import zipfile
import pickle
import jieba
import pandas as pd
import numpy as np
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# input file
ZIP_DATA = os.path.join(ROOT, 'data', '百度题库.zip')  # 要解压的文件
STOPWORDS = os.path.join(ROOT, 'data', 'stopwords.txt')

# output file path
# BERT
TRAIN_TSV = os.path.join(ROOT, 'data', 'train.tsv')  # BERT的数据文件
DEV_TSV = os.path.join(ROOT, 'data', 'dev.tsv')
TEST_TSV = os.path.join(ROOT, 'data', 'test.tsv')

# TextCNN and Transformer
TOKENIZER_BINARIZER = os.path.join(ROOT, 'data', 'tokenizer_binarizer.pickle')
LABELS_FILE = os.path.join(ROOT, 'data', 'label.txt')
X_NPY = os.path.join(ROOT, 'data', 'x.npy')  # testcnn 和 transformer的数据文件
Y_NPY = os.path.join(ROOT, 'data', 'y.npy')


def unzip_data():
    """
    解压数据
    """
    with zipfile.ZipFile(ZIP_DATA, 'r') as z:
        z.extractall(os.path.join(ROOT, 'data'))
        print("已将压缩包解压至{}".format(z.filename.rstrip('.zip')))
        return z.filename.rstrip('.zip')


def combine_data(data_path):
    """
    把四门科目内的所有文件合并
    """
    r = re.compile(r'\[知识点：\]\n(.*)')  # 用来寻找知识点的正则表达式
    r1 = re.compile(r'纠错复制收藏到空间加入选题篮查看答案解析|\n|知识点：|\s|\[题目\]')  # 简单清洗

    data = []
    for root, dirs, files in os.walk(data_path):
        if files:  # 如果文件夹下有csv文件
            for f in files:
                subject = re.findall('高中_(.{2})', root)[0]
                topic = f.strip('.csv')
                tmp = pd.read_csv(os.path.join(root, f))  # 打开csv文件
                tmp['subject'] = subject  # 主标签：科目
                tmp['topic'] = topic  # 副标签：科目下主题
                tmp['knowledge'] = tmp['item'].apply(
                    lambda x: r.findall(x)[0].replace(',', ' ') if r.findall(x) else '')
                tmp['item'] = tmp['item'].apply(lambda x: r1.sub('', r.sub('', x)))
                data.append(tmp)

    data = pd.concat(data).rename(columns={'item': 'content'}).reset_index(drop=True)
    # 删掉多余的两列
    data.drop(['web-scraper-order', 'web-scraper-start-url'], axis=1, inplace=True)
    return data


def extract_label(df, freq=0.01):
    """

    :param df: 合并后的数据集
    :param freq: 要过滤的标签占样本数量的比例
    :return: DataFrame
    """
    knowledges = ' '.join(df['knowledge']).split()  # 合并
    knowledges = Counter(knowledges)
    k = int(df.shape[0] * freq)  # 计算对应频率知识点出现的次数
    print('过滤掉出现次数少于 %d 次的标签' % k)
    top_k = {i for i in knowledges if knowledges[i] > k}  # 过滤掉知识点出现次数小于k的样本
    df.knowledge = df.knowledge.apply(lambda x: ' '.join([label for label in x.split() if label in top_k]))
    df['label'] = df[['subject', 'topic', 'knowledge']].apply(lambda x: ' '.join(x), axis=1)

    return df[['label', 'content']]


def create_bert_data(df, small=False):
    """
    对于 bert 的预处理
    如果small=True：是因为自己的电脑太菜，就用比较小的数据量在本地实现模型
    该函数给bert模型划分了3个数据集
    """
    df['content'] = df['content'].apply(lambda x: x.replace(' ', ''))
    if small:
        print('use small dataset to test my local bert model really work')
        train = df.sample(128)
        dev = df.sample(64)
        test = df.sample(64)
    else:
        train, test = train_test_split(df, test_size=0.2, random_state=2020)
        train, dev = train_test_split(train, test_size=0.2, random_state=2020)

    print('preprocess for bert!')
    print('create 3 tsv file(train, dev, test) in %s' % (os.path.join(ROOT, 'data')))
    train.to_csv(TRAIN_TSV, index=None, sep='\t')
    dev.to_csv(DEV_TSV, index=None, sep='\t')
    test.to_csv(TEST_TSV, index=None, sep='\t')


def load_stopwords():
    return {line.strip() for line in open(STOPWORDS, encoding='UTF-8').readlines()}


def sentence_preprocess(sentence):
    # 去标点
    r = re.compile("[^\u4e00-\u9fa5]+|题目")
    sentence = r.sub("", sentence)  # 删除所有非汉字字符

    # 切词
    words = jieba.cut(sentence, cut_all=False)

    # 去停用词
    stop_words = load_stopwords()
    words = [w for w in words if w not in stop_words]
    return words


def df_preprocess(df):
    """
    合并了去标点，切词，去停用词的操作
    :param df:
    :return:
    """
    df.content = df.content.apply(sentence_preprocess)
    return df


def create_testcnn_data(df, num_words=50000, maxlen=128):
    # 对于label处理
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df.label.apply(lambda label: label.split()))

    with open(LABELS_FILE, mode='w', encoding='utf-8') as f:
        for label in mlb.classes_:
            f.write(label+'\n')

    # 对content处理
    tokenizer = Tokenizer(num_words=num_words, oov_token="<UNK>")
    tokenizer.fit_on_texts(df.content.tolist())
    x = tokenizer.texts_to_sequences(df.content)
    x = pad_sequences(x, maxlen=maxlen, padding='post', truncating='post')  # padding

    # 保存数据
    np.save(X_NPY, x)
    np.save(Y_NPY, y)
    print('已创建并保存x,y至：\n {} \n {}'.format(X_NPY, Y_NPY))

    # 同时还要保存tokenizer和 multi_label_binarizer
    # 否则训练结束后无法还原把数字还原成文本
    tb = {'tokenizer': tokenizer, 'binarizer': mlb}  # 用个字典来保存
    with open(TOKENIZER_BINARIZER, 'wb') as f:
        pickle.dump(tb, f)
    print('已创建并保存tokenizer和binarizer至：\n {}'.format(TOKENIZER_BINARIZER))


def load_testcnn_data():
    """
    如果分开保存，那要保存6个文件太麻烦了。
    所以采取读取之后划分数据集的方式
    """
    # 与之前的bert同步
    x = np.load(X_NPY).astype(np.float32)
    y = np.load(Y_NPY).astype(np.float32)

    # 与之前bert的划分方式统一
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=2020)
    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.2, random_state=2020)

    return train_x, dev_x, test_x, train_y, dev_y, test_y


def load_tokenizer_binarizer():
    """
    读取tokenizer 和 binarizer
    :return:
    """
    with open(TOKENIZER_BINARIZER, 'rb') as f:
        tb = pickle.load(f)
    return tb['tokenizer'], tb['binarizer']


def main():
    """
    合并以上所有操作
    """
    data_path = unzip_data()  # 解压
    df = combine_data(data_path)  # 合并
    df = extract_label(df)  # 提取标签

    # 对于bert的预处理
    create_bert_data(df)

    # 对于testcnn和transformer的预处理
    df = df_preprocess(df)  # 切词，分词，去停用词
    create_testcnn_data(df, num_words=50000, maxlen=128)


if __name__ == '__main__':
    main()
