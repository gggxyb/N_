# -*- coding: utf-8 -*-

# Import the necessary modules
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from utils import *



def stop(train_str):
    # 加载停用词
    stopwords = []
    with open("./MachineLearning/sentiment_analysis/stopwords.txt", "r", encoding="utf8") as f:
        for w in f:
            stopwords.append(w.strip())
    
    train_str_stop = []
    for i in range(len(train_str)):
        train_str_stop.append([w for w in train_str[i] if w not in stopwords])
    return train_str_stop


def metrics_result(actual, predict):
    print(metrics.confusion_matrix(actual, predict))
    print('精度:{0:.3f}'.format(metrics.precision_score(
        actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(
        actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual,
                                                     predict, average='weighted')))
    print(classification_report(actual, predict))

if __name__ == "__main__":
    # 导入字典
    with open('./MachineLearning/sentiment_analysis/lstm/word_dict.pk', 'rb') as f:
        word_dictionary = pickle.load(f)
    with open('./MachineLearning/sentiment_analysis/lstm/label_dict.pk', 'rb') as f:
        output_dictionary = pickle.load(f)

    # try:
    # 数据预处理
    input_shape = 180
    df = pd.read_csv('./MachineLearning/sentiment_analysis/data/test_labled.csv')
    labels, vocabulary = list(df['情感倾向'].unique()), list(df['微博中文内容'].unique())

    test_data = load_corpus("./MachineLearning/sentiment_analysis/data/test_labled.csv")
    df = pd.DataFrame(test_data, columns=["微博中文内容", "情感倾向"])
    df.微博中文内容 = stop(df['微博中文内容'])
    df.微博中文内容 = ["".join(w) for w in df["微博中文内容"]]
    y_label=list(df.情感倾向)
    # sent = "电视刚安装好，说实话，画质不怎么样，很差！"
    # sent = df.微博中文内容
    # x = [[word_dictionary[word] for word in sent]]
    # x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    x = [[word_dictionary[word] for word in sent if word in word_dictionary] for sent in df['微博中文内容']]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    # 载入模型
    model_save_path = 'MachineLearning/sentiment_analysis/lstm/corpus_model.h5'
    lstm_model = load_model(model_save_path)

    # 模型预测
    # metrics_result(labels,lstm_model.predict(x))
    y_predict = lstm_model.predict(x)
    print(y_predict.shape,y_predict)
    label_dict = {v:k for k,v in output_dictionary.items()}
    print(label_dict)
    prediction = []
    p= np.argmax(y_predict,axis=1)
    print(p)
    c=0
    for i in range(len(y_predict)):
        prediction.append(label_dict[p[i]])
        if y_label[i] == prediction[i]:
            c+=1
    print(c)
    metrics_result(y_label,prediction)
    # print('输入语句: %s' % sent)
    # print('情感预测结果: %s' % label_dict[np.argmax(y_predict)])

    # except KeyError as err:
    #     print("您输入的句子有汉字不在词汇表中，请重新输入！")
    #     print("不在词汇表中的单词为：%s." % err)
