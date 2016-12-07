#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy
import sys
from operator import add
import cPickle as pk
from sklearn.metrics.pairwise import cosine_similarity


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
        self.docs = docs
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
        print 'topic init'
        for m, doc in enumerate(docs):
            refresh_output(['initializing', m])
            self.N += len(doc)
            zw_n = []

            # initialize word and images separately
            for t in xrange(self.word_dim):
                if smartinit:
                    p_zw = self.n_zw_t[:, t] * self.n_m_zw[m] / self.n_zw
                    zw = numpy.random.multinomial(1, p_zw / p_zw.sum()).argmax()
                else:
                    zw = numpy.random.randint(0, K)

                zw_n.append(zw)
                self.n_m_zw[m, zw] += 1
                self.n_zw_t[zw, t] += 1
                self.n_zw[zw] += 1
            self.zw_m_n.append(numpy.array(zw_n))

            zi_n = []
            for t in xrange(self.image_dim):
                if smartinit:
                    p_zi = self.n_zi_t[:, t] * self.n_m_zi[m] / self.n_zi
                    zi = numpy.random.multinomial(1, p_zi / p_zi.sum()).argmax()
                else:
                    zi = numpy.random.randint(0, K)

                zi_n.append(zi)
                self.n_m_zi[m, zi] += 1
                self.n_zi_t[zi, t] += 1
                self.n_zi[zi] += 1

            self.zi_m_n.append(numpy.array(zi_n))
        print ''

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            zw_n = self.zw_m_n[m]
            n_m_zw = self.n_m_zw[m]
            zi_n = self.zi_m_n[m]
            n_m_zi = self.n_m_zi[m]

            for t in xrange(self.word_dim):
                # discount for n-th word t with topic z
                zw = zw_n[t]
                n_m_zw[zw] -= 1
                self.n_zw_t[zw, t] -= 1
                self.n_zw[zw] -= 1

                # sampling topic new_z for t
                p_zw = self.n_zw_t[:, t] * n_m_zw / self.n_zw
                new_zw = numpy.random.multinomial(1, p_zw / p_zw.sum()).argmax()

                # set z the new topic and increment counters
                zw_n[t] = new_zw
                n_m_zw[new_zw] += 1
                self.n_zw_t[new_zw, t] += 1
                self.n_zw[new_zw] += 1

            for t in xrange(self.image_dim):
                # discount for n-th word t with topic z
                zi = zi_n[t]
                n_m_zi[zi] -= 1
                self.n_zi_t[zi, t] -= 1
                self.n_zi[zi] -= 1

                # sampling topic new_z for t
                p_zi = self.n_zi_t[:, t] * n_m_zi / self.n_zi
                new_zi = numpy.random.multinomial(1, p_zi / p_zi.sum()).argmax()

                # set z the new topic and increment counters
                zi_n[t] = new_zi
                n_m_zi[new_zi] += 1
                self.n_zi_t[new_zi, t] += 1
                self.n_zi[new_zi] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_zw_t / self.n_zw[:, numpy.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_zw[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= numpy.log(numpy.inner(phi[:, w], theta))
            N += len(doc)
        return numpy.exp(log_per / N)


def lda_learning(lda, iteration, voca=None):
    # pre_perp = lda.perplexity()
    # print ("initial perplexity=%f" % pre_perp)
    print 'topic training'
    for i in range(iteration):
        refresh_output(['training', i])
        print 'training', i
        lda.inference()
        # perp = lda.perplexity()
        # print ("-%d p=%f" % (i + 1, perp))
        # if pre_perp:
        #     if pre_perp < perp:
        #         output_word_topic_dist(lda, voca)
        #         pre_perp = None
        #     else:
        #         pre_perp = perp
        #         output_word_topic_dist(lda, voca)
    print ''
    output_word_topic_dist(lda, voca)


def output_word_topic_dist(lda, top_N=3):
    # find the closest word using the distribution
    tag_name_w2v = pk.load(open('w2v_tags.pk', 'rb'))
    for x in lda.zw_m_n:
        top_words = [''] * top_N
        top_words_sim = [.0] * top_N
        for t, vec in tag_name_w2v.iteritem():
            sim = cosine_similarity([normalize(x), normalize(vec)])[0][1]
            if sim > min(top_words_sim):
                index_to_replace = numpy.array(top_words_sim).argmin()
                top_words[index_to_replace] = t
                top_words_sim[index_to_replace] = sim
        print top_words, top_words

    img_feats = pk.load(open('pic_dict.pk', 'rb'))
    for x in lda.zi_m_n:
        top_words = [''] * top_N
        top_words_sim = [.0] * top_N
        for t, vec in img_feats.iteritem():
            sim = cosine_similarity([normalize(x), normalize(vec)])[0][1]
            if sim > min(top_words_sim):
                index_to_replace = numpy.array(top_words_sim).argmin()
                top_words[index_to_replace] = t
                top_words_sim[index_to_replace] = sim
        print top_words, top_words



def main():
    import optparse
    import vocabulary
    parser = optparse.OptionParser()
    parser.add_option("-f", dest="filename", help="corpus filename", default='complete_document_one_2_one.pk')
    parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
    parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=10)
    parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
    parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true",
                      default=False)
    parser.add_option("--seed", dest="seed", type="int", help="random seed")
    parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
    (options, args) = parser.parse_args()
    if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

    corpus = vocabulary.load_file(options.filename)
    if options.seed != None:
        numpy.random.seed(options.seed)

    lda = LDA(options.K, options.alpha, options.beta, corpus[:1000], [300, 1000], options.smartinit)

    # import cProfile
    # cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
    # output_word_topic_dist(lda, voca)
    lda_learning(lda, options.iteration)


if __name__ == "__main__":
    main()
