import numpy as np
import cPickle as pk
from sklearn.metrics.pairwise import cosine_similarity

n_zw_t = np.load('10t_word_topic.npy')
n_zi_t = np.load('10t_img_topic.npy')

top_N = 10

# tag_name_w2v = pk.load(open('w2v_tags.pk', 'rb'))
# for x in n_zw_t:
#     top_dim = x.argmax()
#
#     tkeys = np.array(tag_name_w2v.keys())
#     tvalues = np.array(tag_name_w2v.values())
#
#     top_N_words_index = np.array(tvalues)[:, top_dim].argsort()
#
#     print tkeys[top_N_words_index][::-1][:top_N], tvalues[:, top_dim][top_N_words_index][::-1][:top_N]

img_feats = pk.load(open('test_pic_dict.pk', 'rb'))
for x in n_zi_t:
    top_dim = x.argmax()

    tkeys = np.array(img_feats.keys())
    tvalues = np.array(img_feats.values())

    top_N_words_index = np.array(tvalues)[:, top_dim].argsort()

    print [str(x) + '-' + str(y) for x, y in
           zip(tkeys[top_N_words_index][::-1][:top_N], tvalues[:, top_dim][top_N_words_index][::-1][:top_N])]
