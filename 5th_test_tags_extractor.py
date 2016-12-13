import glob
import os
import cPickle as pk
from gensim.models.word2vec import Word2Vec as w

file_list = glob.glob('./tags/*')
tag_name_pool = ['flower']
tag_name_w2v = {}

model = w.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
for t in tag_name_pool:
    tag_name_w2v[t] = model[t]

pk.dump(tag_name_w2v, open('test_w2v_tags.pk', 'wb'))
