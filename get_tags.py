import glob
import os
import cPickle as pk

file_list = glob.glob('./tags/*')

pic_dict = {}
for file_name in file_list:
    f = open(file_name, 'r')
    tag_name = file_name.split(os.sep)[-1].split('.')[0]
    for line in f:
        cur_num = int(line.strip())
        if cur_num in pic_dict:
            pic_dict[cur_num].append(tag_name)
        else:
            pic_dict[cur_num] = [tag_name]

pk.dump(pic_dict, open('pic_dict.pk', 'wb'))
