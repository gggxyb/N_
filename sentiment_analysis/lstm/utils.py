# -*- coding: utf-8 -*-
from langconv import *
import pandas as pd
import re
import jieba
import os
import sys
sys.path.insert(0, os.getcwd())


def tokenize(text):
    """
    带有语料清洗功能的分词函数, 包含数据预处理, 可以根据自己的需求重载
    """
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("//@.+?:", " ", text)             # 去除 //@xxx: (用户名)
    text = re.sub("【.+?】", " ", text)             # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("展开全文c", " ", text)           # 去除 '展开全文c'(非用户所写)
    text = re.sub("查看图片", " ", text)             # 去除 '查看图片'(非用户所写)
    icons = re.findall("\[.+?\]", text)             # 提取出所有表情图标
    text = re.sub("\[.+?\]", "IconMark", text)      # 将文本中的图标替换为`IconMark`
    tokens = []
    for k, w in enumerate(jieba.lcut(text)):
        w = w.strip()
        if "IconMark" in w:                         # 将IconMark替换为原图标
            for i in range(w.count("IconMark")):
                tokens.append(icons.pop(0))
        elif w and w != '\u200b' and w.isalpha():   # 只保留有效文本
            tokens.append(Traditional2Simplified(w))
    return tokens


def load_corpus(path):
    """
    加载语料库
    """
    data = pd.read_csv(path)
    data = data.values.tolist()
    for i in range(len(data)):
        data[i][0] = tokenize(data[i][0])             # 分词
    return data


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence
