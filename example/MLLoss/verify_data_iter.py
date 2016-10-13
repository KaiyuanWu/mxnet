# -*- coding: utf-8 -*-
import os
import cv2
import leveldb
import logging
import numpy as np
import ImageAugmenterPy
import struct


class VerifyDataIter():
    def __init__(self, leveldb_dir_pos, leveldb_dir_neg,  batch_size_pos, batch_size_neg, random_skip,  aux_args):
	try:
	    self.leveldb_pos = leveldb.LevelDB(leveldb_dir_pos)
	    self.dataiter_pos = self.leveldb_pos.RangeIter(include_value = False)
	    self.leveldb_neg = leveldb.LevelDB(leveldb_dir_neg)
	    self.dataiter_neg = self.leveldb_neg.RangeIter(include_value = False)
	    
	except Exception as e:
	    logging.error("Fail to open leveldb %s %s"%(leveldb_dir_pos, leveldb_dir_neg))
	    logging.error("Error message: %s", e.message)
	    quit()
	    
	self.batch_size_pos = batch_size_pos
	self.batch_size_neg = batch_size_neg
	self.random_skip = random_skip
	
	self.num_aug = 1
	if 'num_aug' in aux_args:
	    self.num_aug = aux_args['num_aug']
	if 'data_shape' not in aux_args:
	    logging.error("No data shape provided in aux_args")
	    quit()
	else:
	    self.c = aux_args['data_shape'][0]
	    self.h = aux_args['data_shape'][1]
	    self.w = aux_args['data_shape'][2]
	if 'with_additional_label' not in aux_args:
	    aux_args['with_additional_label'] = False
	self.aux_args = aux_args    
	self.read_person_pos = 0
	self.read_img_pos = 0
	self.read_img_neg = 0
	
	max_imgs_per_person = 300
	self.data_buf = np.empty(((max_imgs_per_person*batch_size_pos + batch_size_neg)*self.num_aug, self.c, self.h, self.w), dtype = 'float32')
	self.label_buf = np.empty(((max_imgs_per_person*batch_size_pos + batch_size_neg)*self.num_aug), dtype = 'int')
	self.pos_set = np.zeros(batch_size_pos, dtype = 'int')
       
    def provide_data(self):
        return [('data',(self.read_img_pos + self.read_img_neg, self.c, self.h, self.w))]   
    
    def provide_label(self):
        return [('label',(self.read_img_pos + self.read_img_neg)), ('pos_set', (self.read_person_pos))]
    
    def reset(self):
	self.read_person_pos = 0
	self.read_img_pos = 0
	self.read_img_neg = 0
        try:
	    self.dataiter_pos = self.leveldb_pos.RangeIter(include_value = False)
	    self.dataiter_neg = self.leveldb_neg.RangeIter(include_value = False)
	except Exception as e:
	    logging.error("Fail to reset leveldb  dataiter for %s %s"%(self.leveldb_dir_pos, self.leveldb_dir_neg))
	    logging.error("Error message: %s", e.message)
	    quit()
	    
    def dataiter_pos_next(self):
	#试着读取一个人的所有照片，已经假设pos leveldb中的照片是按人连续存储的
	try:
	    key = self.dataiter_pos.next()
	except Exception as e:
	    self.dataiter_pos = self.leveldb_pos.RangeIter(include_value = False)
	    key = self.dataiter_pos.next()
	return key   
    
    def dataiter_neg_next(self):
	try:
	    key = self.dataiter_neg.next()
	except Exception as e:
	    self.dataiter_neg = self.leveldb_neg.RangeIter(include_value = False)
	    key = self.dataiter_neg.next()
	return key 
    
    def get_batch(self):
        self.read_person_pos = -1
	self.read_img_pos = 0
	self.read_img_neg = 0
	
	#读取正样本
	prev_idx = -1
	key = self.dataiter_pos_next()
	pos_set = {}
	while 1:
	    #随机跳过 random_skip个样本
	    step = np.random.randint(0, self.random_skip + 1)
	    istep = 0
	    while istep < step:
		key = self.dataiter_pos_next()
		istep += 1    
	    index, idx, fmt = key.split('_')
	    idx = int(idx)
	    if idx != prev_idx:
		self.read_person_pos += 1
		prev_idx = idx
		
	    if 	self.read_person_pos >= self.batch_size_pos:
		break
	    else:
		self.pos_set[self.read_person_pos] = idx
		pos_set[idx] = 1
		
	    value = self.leveldb_pos.Get(key)
	    if self.aux_args['with_additional_label']:
		id, img_str,h,w,c, addtional_label_str, addtional_label_length, image_filename = struct.unpack(fmt, value)
	    else:
		id, img_str,h,w,c,  image_filename = struct.unpack(fmt, value)
	    
	    tmp_img = np.zeros((h,w,c), dtype='uint8')
	    tmp_img.data[:] = img_str
#	    cv2.imshow("tmp_img", tmp_img)
#	    cv2.waitKey(0)
	    for iaug_img  in range(self.num_aug):
		aug_img = ImageAugmenterPy.augment_img_process(tmp_img,args=self.aux_args)
		#将矩阵的shape从opencv的[h,w,c]变成深度学习里面的[c,h,w]
		self.data_buf[self.read_img_pos] = np.rollaxis(aug_img.astype('float32'),2,0)
		self.label_buf[self.read_img_pos] = idx
		self.read_img_pos += 1
	    	
	#读取负样本
	key = self.dataiter_neg_next()
	while self.read_img_neg < self.batch_size_neg:
	    #随机跳过 random_skip个样本
	    step = np.random.randint(0, self.random_skip + 1)
	    istep = 0
	    while istep < step:
		key = self.dataiter_neg_next()
		istep += 1    
	    index, idx, fmt = key.split('_')
	    idx = int(idx)
	    if idx in pos_set:
		continue
	    
	    value = value = self.leveldb_neg.Get(key)
	    if self.aux_args['with_additional_label']:
		id, img_str,h,w,c, addtional_label_str, addtional_label_length, image_filename = struct.unpack(fmt, value)
	    else:
		id, img_str,h,w,c,  image_filename = struct.unpack(fmt, value)
	    tmp_img = np.zeros((h,w,c), dtype='uint8')
	    tmp_img.data[:] = img_str
	    for iaug_img  in range(self.num_aug):
		aug_img = ImageAugmenterPy.augment_img_process(tmp_img,args=self.aux_args)
		#将矩阵的shape从opencv的[h,w,c]变成深度学习里面的[c,h,w]
		self.data_buf[self.read_img_pos + self.read_img_neg] = np.rollaxis(aug_img.astype('float32'),2,0)
		self.label_buf[self.read_img_pos+ self.read_img_neg] = idx
		self.read_img_neg += 1
	
	return {'data': self.data_buf[:self.read_img_pos + self.read_img_neg],'label':self.label_buf[:self.read_img_pos + self.read_img_neg],
		'pos_set': self.pos_set}
		
