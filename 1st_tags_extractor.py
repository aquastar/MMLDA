import glob
import os
import cPickle as pk
from gensim.models.word2vec import Word2Vec as w

file_list = glob.glob('./tags/*')
tag_name_pool = set()
tag_name_w2v = {}

pic_tags = {}
for file_name in file_list:
    f = open(file_name, 'r')
    tag_name = file_name.split(os.sep)[-1].split('.')[0].split('_')[0]
    tag_name_pool.add(tag_name)
    for line in f:
        cur_num = int(line.strip())
        if cur_num in pic_tags:
            pic_tags[cur_num].append(tag_name)
        else:
            pic_tags[cur_num] = [tag_name]

pk.dump(pic_tags, open('pic_tags.pk', 'wb'))

model = w.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
for t in tag_name_pool:
    tag_name_w2v[t] = model[t]

pk.dump(tag_name_w2v, open('w2v_tags.pk', 'wb'))
