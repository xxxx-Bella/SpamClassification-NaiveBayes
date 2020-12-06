# -*- coding: UTF-8 -*-
# 垃圾邮件过滤-朴素贝叶斯实现
import numpy as np
import re
import random


def createVocabList(dataSet):
    '''
    将切分的实验样本词条整理成不重复的词条列表，即词汇表
    param:
        dataSet - 整理的样本数据集
    return:
        vocabulary - 返回不重复的词条列表，也就是词汇表
    '''
    vocabulary = set([])  # 使用集合，方便去重
    for document in dataSet:
        vocabulary = vocabulary | set(document)  # 取并集
    return list(vocabulary)


def Words2Vec(vocabList, inputSet):
    """
    根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    param:
        vocabList - createVocabList()返回的词汇列表
        inputSet - 待切分的词条列表
    return:
        docVec - 文档向量，词集模型
    """
    docVec = [0] * len(vocabList) #创建一个词汇表大小的全0向量
    for word in inputSet: # 遍历每个词条
        if word in vocabList: # 如果该词条存在于词汇表中，则置为1
            docVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return docVec   #返回文档向量


def trainNB(X_train, y_train):
    """
    朴素贝叶斯分类器训练函数
    param:
        X_train - 训练文档矩阵，即Words2Vec返回的docVec构成的矩阵
        y_train - 训练类别标签向量，即loadDataSet返回的classVec
    return:
        p0Vect - 正常邮件类的条件概率数组
        p1Vect - 垃圾邮件类的条件概率数组
        pSpam - 文档属于垃圾邮件类的概率
    """
    trainDoc_num = len(X_train)  # 计算训练集文档数目(=40)
    word_num = len(X_train[0])  # 计算每篇文档的词条数
    #print(word_num)
    pSpam = sum(y_train) / float(trainDoc_num)  # 文档属于垃圾邮件类的概率
    p0Num = np.ones(word_num)
    p1Num = np.ones(word_num)  # 词条出现次数初始化为1（拉普拉斯平滑）
    p0Denom = 2.0 
    p1Denom = 2.0  # 分母初始化为2（拉普拉斯平滑）
    for i in range(trainDoc_num):
        if y_train[i] == 1:  # 统计属于垃圾类的条件概率所需的数据，即P(wi|1)
            p1Num += X_train[i]
            p1Denom += sum(X_train[i])
        else:  # 统计属于正常类的条件概率所需的数据，即P(wi|0)
            p0Num += X_train[i]
            p0Denom += sum(X_train[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom) # 取对数，防止下溢出
    return p0Vect, p1Vect, pSpam


def NBClassifier(wordArray, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类器分类器
    param:
	    wordArray - 待分类的词条数组
	    p0Vec - 正常邮件类的条件概率数组
	    p1Vec - 垃圾邮件类的条件概率数组
	    pClass1 - 文档属于垃圾邮件的概率
    return:
	    0 - 正常邮件
	    1 - 垃圾邮件
    """
    p1 = sum(wordArray*p1Vec)+np.log(pClass1)
    p0 = sum(wordArray*p0Vec)+np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def Str2List(string):
    '''
    将字符串转换为字符列表
    '''
    token_list = re.split(r'\W+', string)  # 以特殊符号（非字母、非数字）作为切分标志进行字符串切分
    return [tk.lower() for tk in token_list if len(tk) > 2]  # 除了单个字母，例如大写的I，其它单词变成小写


def load_data():
    '''
    读入数据，返回邮件内容列表和对应的类别列表
    return:
    docList - 邮件内容
    classList - 邮件类别
    '''
    docList = [] 
    classList = [] 
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = Str2List(open('email/spam/%d.txt' % i, 'r').read())# 读取每个垃圾邮件，将字符串转换成字符串列表
        docList.append(wordList)
        classList.append(1)  # 标记垃圾邮件，1为垃圾邮件
        wordList = Str2List(open('email/ham/%d.txt' % i, 'r').read())# 读取每个非垃圾邮件，将字符串转换成字符串列表
        docList.append(wordList)
        classList.append(0)  # 标记正常邮件，0为正常邮件
    # print('docList: ',docList)
    # print('classList: ',classList)
    return docList, classList


def OneCrossValidate(X_train, y_train, X_test, y_test):
    '''
    一次交叉验证
    param:
        X_train - 训练文档矩阵
        y_train - 训练类别标签向量
        X_test - 验证集文档矩阵
        y_test - 验证集类别标签向量
    return:
        error_rate - 分类错误率
    '''
    p0V, p1V, pSpam = trainNB(np.array(X_train), np.array(y_train))  
    errorCount = 0  # 错误分类计数器
    for id in range(10):  # 遍历验证集进行测试
        if NBClassifier(np.array(X_test[id]), p0V, p1V, pSpam) != y_test[id]:  # 如果分类错误
            errorCount += 1
    error_rate = errorCount/10
    return(error_rate)


def KFold(trainDocList, trainClassList,k):
    '''
    K折交叉验证
    param:
        trainDocList - 所有文档词汇向量矩阵
        trainClassList - 文档类别
        k - k_fold
    return：
        error_radio - 分类错误率
    '''
    randInd = list(range(50))
    random.shuffle(randInd)
    error_radio = 0.0
    for i in range(k):
        rdInd = randInd # 随机索引
        # 选取训练集、验证集索引
        X_train_id = set(randInd[10*i:10*i+10])  # 训练集
        X_test_id = set(rdInd) - X_train_id  # 验证集
        # 选取训练集、验证集数据
        X_train, y_train, X_test, y_test = [],[],[],[]
        for idx in X_train_id:
            X_train.append(trainDocList[idx])
            y_train.append(trainClassList[idx])
        # 遍历验证集
        for idx in X_test_id:
            X_test.append(trainDocList[idx])
            y_test.append(trainClassList[idx])
        error_radio += OneCrossValidate(X_train,y_train, X_test, y_test)
        error_radio = error_radio/k
    return error_radio


def testSpam1():
    '''
    测试朴素贝叶斯分类器，不进行交叉验证，随机划分训练集和测试集
    '''
    docList, classList = load_data() # 读入数据
    vocabList = createVocabList(docList)  # 创建词汇表
    train_id = list(range(50)) # 存储训练集的索引值的列表
    test_id = []  # 存储测试集的索引值的列表
    #划分训练集和测试集
    for i in range(10):  # 从总共的50个邮件中，随机挑选出40个作为训练集，剩余10个作为测试集
        rand_id = int(random.uniform(0, len(train_id)))  # 随机选取索引值
        test_id.append(train_id[rand_id])
        del(train_id[rand_id])
    X_train = [] # 训练集矩阵
    y_train = []  # 训练集类别标签向量
    # 遍历训练集
    for id in train_id:  
        docVec = Words2Vec(vocabList, docList[id])
        X_train.append(docVec)
        y_train.append(classList[id])
    # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB(np.array(X_train), np.array(y_train))  
    errorCount = 0  # 错误分类计数器
    for id in test_id:  # 遍历测试集
        wordVector = Words2Vec(vocabList, docList[id])
        if NBClassifier(np.array(wordVector), p0V, p1V, pSpam) != classList[id]:  # 如果分类错误
            errorCount += 1
            # print("分类错误的测试集：", docList[id])
    error_rate = float(errorCount) / len(test_id)
    return error_rate


def testSpam2(k):
    '''
    测试朴素贝叶斯分类器，进行交叉验证
    '''
    docList, classList = load_data() # 读入数据
    vocabList = createVocabList(docList)  # 创建词汇表
    # 获得所有文档词汇向量矩阵
    all_DocList = [] 
    for doc in docList:
        all_DocList.append(Words2Vec(vocabList, doc))
    error_rate = KFold(all_DocList, classList, k) # 5折交叉验证
    return error_rate


if __name__ == '__main__':
    iter_error_rate1 = []
    iter_error_rate2 = []
    n, k= 10, 5
    sum_err1, sum_err2 = 0.0, 0.0
    #ave1, ave2 = 
    for i in range(n):
        er1 = testSpam1()
        er2 = testSpam2(k)
        err1 = "%.2f%%" % (er1 * 100) # 转化为百分制
        err2 = "%.2f%%" % (er2 * 100)
        iter_error_rate1.append(err1)
        iter_error_rate2.append(err2)
        sum_err1 += er1
        sum_err2 += er2
        ave1 = "%.2f%%" % (sum_err1/n * 100)
        ave2 = "%.2f%%" % (sum_err2/n * 100)
    print('1.进行 {} 次循环，分类错误率：{}\n平均错误率为：{}'.format(n, iter_error_rate1, ave1))
    print('2.进行 {} 次 {} 折交叉验证，分类错误率：{}\n平均错误率为：{}'.format(n, k, iter_error_rate2, ave2))