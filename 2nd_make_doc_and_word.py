import cPickle as pk
import os

# treat the dimension as the hidden 'words'
# one document includes one pic and one tag, multiple tags will be decomposed into 1 pic corresponding to 1 tag name

# merge using the tag record
img_tag_pair = pk.load(open('pic_tags.pk', 'rb'))
tag_name_w2v = pk.load(open('w2v_tags.pk', 'rb'))
img_feats = pk.load(open('pic_dict.pk', 'rb'))

complete_document = []
for img_prefix, tags in img_tag_pair.iteritems():
    print img_prefix, tags
    actual_document = []
    for t in tags:
        actual_document.append(tag_name_w2v[t])
        actual_document.append(img_feats['./mirflickr/im%s.jpg' % (img_prefix)])

        complete_document.append(actual_document)

pk.dump(complete_document, open('complete_document_one_2_one.pk', 'wb'))
