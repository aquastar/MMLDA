import os

import ListUtil
import LoadData
import Preprocess
from Sample import UniSample, MultSample


class LDAModel:
    alpha = float
    beta = float
    D = int
    K = int
    W = int
    Top_Words_Num = int
    NumberOfIterations = int
    SaveStep = int

    Dictionary = object
    Z = object
    W = object
    IDListSet = object
    nw = object
    nd = object
    nwsum = object
    ndsum = object
    theta = object
    phi = object

    def __init__(self, alpha, beta, NumberOfIterations, SaveStep, K, TopNum):
        self.alpha = alpha
        self.beta = beta
        self.NumberOfIterations = NumberOfIterations
        self.SaveStep = SaveStep
        self.K = K
        self.nwsum = ListUtil.Initial(self.K)
        self.Top_Words_Num = TopNum

    def ModelInit(self, filename):
        Docs = LoadData.LoadDataFromFile(os.getcwd() + "/" + filename)
        self.D = len(Docs)
        print "Load ", self.D, " docs from the file"
        StopWordList = LoadData.LoadStopWords()
        WordListSet = [Preprocess.PreprocessText(doc, StopWordList) for doc in Docs if type(doc) != unicode]
        self.Dictionary = Preprocess.ConstructDictionary(WordListSet)
        self.W = len(self.Dictionary)
        print "Total number of words is: ", self.W
        print "Begin to save the dictionary..."
        self.SaveDictionary()
        print "Done!!"
        print "Begin to map the word to ID"
        self.IDListSet = []
        inv_dict = {v: k for k, v in self.Dictionary.iteritems()}
        for wdl in WordListSet:
            IdList = Preprocess.Word2Id(wdl, inv_dict)
            self.IDListSet.append(IdList)
        print "Done!!"
        self.ndsum = ListUtil.Initial(self.D)
        self.theta = ListUtil.InitialMat(self.D, self.K, 0.0)
        self.phi = ListUtil.InitialMat(self.K, self.W, 0.0)
        self.nd = ListUtil.InitialMat(self.D, self.K, 0)
        self.nw = ListUtil.InitialMat(self.W, self.K, 0)
        self.Z = []
        print "Begin to initialize the LDA model..."
        self.RandomAssignTopic()
        print "Topic assignment done!!"

    def SaveDictionary(self):
        path = os.getcwd() + "/dictionary.txt"
        fp = open(path, 'w')
        for k, v in self.Dictionary.iteritems():
            fp.write(str(k) + '\t' + v)
        fp.close()

    def RandomAssignTopic(self):
        for d in xrange(self.D):
            DocSize = len(self.IDListSet[d])
            row = ListUtil.Initial(DocSize)
            self.Z.append(row)
            for w in xrange(DocSize):
                topic = UniSample(self.K)
                wid = self.IDListSet[d][w]
                self.Z[d][w] = topic
                self.nw[wid][topic] += 1
                self.nd[d][topic] += 1
                self.nwsum[topic] += 1
            self.ndsum[d] = DocSize

    def sampling(self, d, w):
        topic = self.Z[d][w]
        wid = self.IDListSet[d][w]
        self.nw[wid][topic] -= 1
        self.nd[d][topic] -= 1
        self.nwsum[topic] -= 1
        self.ndsum[d] -= 1

        p = self.ComputeTransProb(d, w)

        newtopic = MultSample(p)
        self.nw[wid][newtopic] += 1
        self.nd[d][newtopic] += 1
        self.nwsum[newtopic] += 1
        self.ndsum[d] += 1
        return newtopic

    def ComputeTransProb(self, d, w):
        Wbeta = self.W * self.beta
        Kalpha = self.K * self.alpha
        wid = self.IDListSet[d][w]
        p = ListUtil.Initial(self.K, 0.0)
        for k in xrange(self.K):
            p[k] = (float(self.nw[wid][k]) + self.beta) / (float(self.nwsum[k]) + Wbeta) * (
                float(self.nd[d][k]) + self.alpha) / (float(self.ndsum[d]) + Kalpha)
            return p

    def ComputeTheta(self):
        for d in xrange(self.D):
            for k in xrange(self.K):
                self.theta[d][k] = (float(self.nd[d][k]) + self.alpha) / (float(self.ndsum[d]) + self.K * self.alpha)

    def ComputePhi(self):
        for k in xrange(self.K):
            for w in xrange(self.W):
                self.phi[k][w] = (self.nw[w][k] + self.beta) / (self.nwsum[k] + self.W * self.beta)

    def SaveTempRes(self, prefix):
        print 'current iteration result:', prefix
        for x in xrange(self.K):
            print '----'
            print 'topic K:', x
            twords = []
            twords = [(n, self.phi[x][n]) for n in xrange(self.W)]
            twords.sort(key=lambda i: i[1], reverse=True)
            for y in xrange(self.Top_Words_Num):
                word = self.Dictionary[twords[y][0]]
                print  word,
            print '----'

    def estimate(self):
        for i in xrange(1, self.NumberOfIterations + 1):
            for d in xrange(self.D):
                for w in xrange(len(self.IDListSet[d])):
                    ListUtil.refresh_output(['iteration:', i, 'document:', d, 'word:', w])
                    newtopic = self.sampling(d, w)
                    self.Z[d][w] = newtopic
            if i % self.SaveStep == 0:
                self.ComputeTheta()
                self.ComputePhi()
                self.SaveTempRes(i)
