#coding:utf-8

import numpy as np
import os
import gzip
from six.moves import urllib
import operator
from datetime import datetime

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#����mnist���ݼ�������tensorflow��base.py�е�д����
def maybe_download(filename, path, source_url):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
    return filepath

#��32λ��ȡ����ҪΪ��У���롢ͼƬ�������ߴ�׼����
#����tensorflow��mnist.pyд�ġ�
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#��ȡͼƬ�����������󣬿ɽ�ͼƬ�еĻҶ�ֵ��ֵ�����������󣬿ɽ���ֵ��������ݴ�ɾ����������
#����tensorflow��mnist.pyд��
def extract_images(input_file, is_value_binary, is_matrix):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print (magic, num_images, rows, )
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return np.minimum(data, 1)
        else:
            return data


#��ȡ��ǩ
#����tensorflow��mnist.pyд��
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

# һ���knn���࣬��ȫ������ͬʱ����һ����룬Ȼ���ҳ���С�����k��ͼ�����ҳ���k��ͼƬ�ı�ǩ����ǩռ������ΪnewInput��label
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row
    init_shape = newInput.shape[0]
    newInput = newInput.reshape(1, init_shape)
    #np.tile(A,B)���ظ�A B�Σ��൱���ظ�[A]*B
    #print np.tile(newInput, (numSamples, 1)).shape
    diff = np.tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)

    classCount = {} # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]

        ## step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    ## step 5: the max voted class will return
    maxCount = 0
    maxIndex = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


maybe_download('train_images', 'data/mnist', SOURCE_URL+TRAIN_IMAGES)
maybe_download('train_labels', 'data/mnist', SOURCE_URL+TRAIN_LABELS)
maybe_download('test_images', 'data/mnist', SOURCE_URL+TEST_IMAGES)
maybe_download('test_labels', 'data/mnist', SOURCE_URL+TEST_LABELS)



# ���������ȶ�ͼƬ��Ȼ�����ڲ�����д����
def testHandWritingClass():
    ## step 1: load data
    print ("step 1: load data...")
    train_x = extract_images('data/mnist/train_images', True, True)
    train_y = extract_labels('data/mnist/train_labels')
    test_x = extract_images('data/mnist/test_images', True, True)
    test_y = extract_labels('data/mnist/test_labels')

    ## step 2: training...
    print ("step 2: training...")
    pass

    ## step 3: testing
    print ("step 3: testing...")
    a = datetime.now()
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = numTestSamples/10
    for i in range(int(test_num)):

        predict = kNNClassify(test_x[i], train_x, train_y, 3)
        if predict == test_y[i]:
            matchCount += 1
        if i % 100 == 0:
            print ("finish%dpicture"%(i))
    accuracy = float(matchCount) / test_num
    b = datetime.now()
    print ("total %d sec"%((b-a).seconds))

    ## step 4: show the result
    print ("step 4: show the result...")
    print ('The classify accuracy is: %.2f%%' % (accuracy * 100))

if __name__ == '__main__':
    testHandWritingClass()
