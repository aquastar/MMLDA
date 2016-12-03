import cPickle as pk
# treat the dimension as the hidden 'words'
# one document includes one pic and one tag, multiple tags will be decomposed into 1 pic corresponding to 1 tag name

# merge using the tag record
tags = pk.load(open('pic_tags.pk', 'rb'))
for img_prefix, tag in tags.iteritems():
    pass