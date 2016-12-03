import glob
import cPickle as pk
import vgg

net = vgg.build_convnet()
file_list = glob.glob('./mirflickr/*')
pic_dict = {}
feats = vgg.compute_fromfile(net, file_list)

for fid, f in enumerate(file_list):
    pic_dict[f] = feats[fid]

pk.dump(pic_dict, open('pic_dict.pk', 'wb'))
