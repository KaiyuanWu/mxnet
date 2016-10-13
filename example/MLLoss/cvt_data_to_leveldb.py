# -*- coding: utf-8 -*-
# 将图片数据存储到leveldb中
import cv2
import numpy as np
import leveldb
import logging
import struct
import sys
FORMAT = '%(asctime)-15s  %(message)s'
logging.basicConfig(format=FORMAT)

__author__ = "kaiwu"
__date__ = "$Oct 13, 2016 11:24:32 AM$"

#filename是一个列表文件，里面包含需要存储的图片列表     
#idx id image_filename additional_label
#example: 
def cvt_data_to_leveldb(filename,  leveldb_dir, addtional_label_length=0):
    db = leveldb.LevelDB(leveldb_dir,write_buffer_size = 536870912)
    index = 0
    for line in open(filename):
        fields = line.split()
        id = int(fields[0])
        image_filename = fields[1]
        if addtional_label_length > 0:
            addtional_label = np.array([float(i) for i in fields[2:]], dtype = 'float32')
        try:
            img = cv2.imread(image_filename, -1)
	    if len(img.shape)==2:
		c = 1
		h,w = img.shape
	    else:
		h,w,c = img.shape
            img_str = img.tostring()
            if addtional_label_length > 0:
                addtional_label_str = addtional_label.tostring()
                fmt="i%dsiii%dsi%ds"%(len(img_str),len(addtional_label_str),len(image_filename))
                value = struct.pack(fmt, id, img_str,h,w,c, addtional_label_str,addtional_label_length, image_filename)
            else:
                fmt="i%dsiii%ds"%(len(img_str),len(image_filename))
                value = struct.pack(fmt, id, img_str,h,w,c,  image_filename)
            key = "%012d_%010d_%s"%(index, id, fmt)
            db.Put(key, value)
        except Exception as e:
            logging.error("Error while save %s, error: %s"%(image_filename, e.message))
        if index%10000 == 0:
            logging.warning("processed %d items"%index)
        index += 1    
    logging.warning("finish process all items")

#example: python cvt_data_to_leveldb.py /tmp/mnist/train_list.txt /tmp/mnist/train_leveldb 0 
#       
filename = sys.argv[1]
leveldb_dir = sys.argv[2]
addtional_label_length = int(sys.argv[3])
cvt_data_to_leveldb(filename,  leveldb_dir, addtional_label_length)    
