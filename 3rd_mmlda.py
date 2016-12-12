#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
import sys
from random import shuffle, sample
from scipy.misc import logsumexp
from math import e
from operator import add
import cPickle as pk
import time
from sklearn.metrics.pairwise import cosine_similarity


def log_softmax(vec):
    return vec - logsumexp(vec)


def softmax(vec):
    return numpy.exp(log_softmax(vec))


def refresh_output(data):
    to_print = ' '.join([str(x) for x in data]) if isinstance(data, list) else data
    sys.stdout.write('%s\r' % to_print)
    sys.stdout.flush()


def normalize(x):
    max_x = max(x)
    min_x = min(x)
    delta = max_x - min_x
    return [(xx - min_x) / delta for xx in x]


class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True):
        self.word_dim = 300
        self.image_dim = 1000
        self.K = K
        self.alpha = alpha  # parameter of topics prior
        self.beta = beta  # parameter of words prior
        self.docs = numpy.array(docs) + 1
        self.V = 0
        self.Vw = V[0]
        self.Vi = V[1]

        # topic with words
        self.zw_m_n = []  # topics of words of documents
        self.n_m_zw = numpy.zeros((len(self.docs), K)) + alpha  # word count of each document and topic
        self.n_zw_t = numpy.zeros((K, self.Vw)) + beta  # word count of each topic and vocabulary
        self.n_zw = numpy.zeros(K) + self.Vw * beta  # word count of each topic

        self.N = 0

        # topic with images
        self.zi_m_n = []  # topics of images of documents
        self.n_m_zi = numpy.zeros((len(self.docs), K)) + alpha  # word count of each document and topic
        self.n_zi_t = numpy.zeros((K, self.Vi)) + beta  # word count of each topic and vocabulary
        self.n_zi = numpy.zeros(K) + self.Vi * beta  # word count of each topic

        # topic initialization
        docs = self.docs
        print 'topic init'
        for m, doc in enumerate(docs):
            refresh_output(['initializing', m])
            self.N += len(doc)

            # initialize word and images separately
            zw_n = []
            for t in xrange(self.word_dim):
                if smartinit:
                    p_zw = self.n_zw_t[:, t] * self.n_m_zw[m] / self.n_zw
                    zw = numpy.random.multinomial(1, p_zw / p_zw.sum()).argmax()
                else:
                    zw = numpy.random.randint(0, K)
                discount_word_weight = doc[0][t]

                zw_n.append(zw)
                self.n_m_zw[m, zw] += discount_word_weight
                self.n_zw_t[zw, t] += discount_word_weight
                self.n_zw[zw] += discount_word_weight
            self.zw_m_n.append(numpy.array(zw_n))

            zi_n = []
            for t in xrange(self.image_dim):
                if smartinit:
                    p_zi = self.n_zi_t[:, t] * self.n_m_zi[m] / self.n_zi
                    zi = numpy.random.multinomial(1, p_zi / p_zi.sum()).argmax()
                else:
                    zi = numpy.random.randint(0, K)

                discount_word_weight = doc[1][t]

                zi_n.append(zi)
                self.n_m_zi[m, zi] += discount_word_weight
                self.n_zi_t[zi, t] += discount_word_weight
                self.n_zi[zi] += discount_word_weight
            self.zi_m_n.append(numpy.array(zi_n))
        print ''

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(sample(self.docs, 100)):
            refresh_output(['current doc', m])
            zw_n = self.zw_m_n[m]
            n_m_zw = self.n_m_zw[m]
            zi_n = self.zi_m_n[m]
            n_m_zi = self.n_m_zi[m]

            for t in xrange(self.word_dim):
                # discount for n-th word t with topic z
                discount_word_weight = doc[0][t]

                zw = zw_n[t]
                n_m_zw[zw] -= discount_word_weight
                self.n_zw_t[zw, t] -= discount_word_weight
                self.n_zw[zw] -= discount_word_weight

                # sampling topic new_z for t
                p_zw = self.n_zw_t[:, t] * n_m_zw / self.n_zw
                new_zw = numpy.random.multinomial(1, softmax(p_zw / p_zw.sum())).argmax()

                # set z the new topic and increment counters
                zw_n[t] = new_zw
                n_m_zw[new_zw] += discount_word_weight
                self.n_zw_t[new_zw, t] += discount_word_weight
                self.n_zw[new_zw] += discount_word_weight

            for t in xrange(self.image_dim):
                # discount for n-th word t with topic z
                discount_word_weight = doc[1][t]

                zi = zi_n[t]
                n_m_zi[zi] -= discount_word_weight
                self.n_zi_t[zi, t] -= discount_word_weight
                self.n_zi[zi] -= discount_word_weight

                # sampling topic new_z for t
                p_zi = self.n_zi_t[:, t] * n_m_zi / self.n_zi
                new_zi = numpy.random.multinomial(1, softmax(p_zi / p_zi.sum())).argmax()

                # set z the new topic and increment counters
                zi_n[t] = new_zi
                n_m_zi[new_zi] += discount_word_weight
                self.n_zi_t[new_zi, t] += discount_word_weight
                self.n_zi[new_zi] += discount_word_weight

    def worddist(self):
        """get topic-word distribution"""
        return self.n_zw_t / self.n_zw[:, numpy.newaxis], self.n_zi_t / self.n_zi[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None:
            docs = self.docs
        phi = self.worddist()
        log_per_1 = 0
        log_per_2 = 0
        N_1 = 0
        N_2 = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(numpy.array(docs)[:, 0]):
            theta = self.n_m_zw[m] / (sum(doc) + Kalpha)
            for w in xrange(len(doc)):
                log_per_1 -= numpy.log(numpy.inner(phi[0][:, w], theta))
            N_1 += sum(doc)

        for m, doc in enumerate(numpy.array(docs)[:, 1]):
            theta = self.n_m_zi[m] / (sum(doc) + Kalpha)
            for w in xrange(len(doc)):
                log_per_2 -= numpy.log(numpy.inner(phi[1][:, w], theta))
            N_2 += sum(doc)
        return numpy.exp(log_per_1 / N_1), numpy.exp(log_per_2 / N_2)


def lda_learning(lda, iteration, voca=None):
    print 'topic training'
    # start = time.time()

    for i in range(iteration):
        if i == 0:
            print 'init topic distribution'
            output_word_topic_dist(lda, str(i))

        lda.inference()

        if i % 50 == 0:
            perp = lda.perplexity()
            print ("-%d p=%f, %f" % (i + 1, perp[0], perp[1]))

    print 'final topic distribution'
    output_word_topic_dist(lda, str(i))


def output_word_topic_dist(lda, prefix):
    numpy.save(prefix + '_word_topic.npy', lda.n_zw_t)
    numpy.save(prefix + '_img_topic.npy', lda.n_zi_t)


def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename", default='complete_document_one_2_one.pk')
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.1)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.1)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=10)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=500)
    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=True)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true",
                      default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    corpus = vocabulary.load_file(options.filename)
    if options.seed != None:
        numpy.random.seed(options.seed)

    lda = LDA(options.K, options.alpha, options.beta, corpus, [300, 1000], options.smartinit)
    lda_learning(lda, options.iteration)


if __name__ == "__main__":
    main()
