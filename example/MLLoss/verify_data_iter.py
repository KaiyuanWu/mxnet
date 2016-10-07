# -*- coding: utf-8 -*-
import mxnet as mx
import random
import os
import numpy as np

class MajorDataIter(mx.io.DataIter):
    def __init__(self, data_dir, num_genuine = 1, num_imposter = 3, img_size=(3, 32, 32), shuffle=False):
        super(MajorDataIter, self).__init__()
        self.data_dir = data_dir
        self.num_genuine = num_genuine
        self.num_imposter = num_imposter
        self.img_size = img_size
        
        self.samples = os.listdir(data_dir)
        self.num_classes = len(self.samples)
        self.data = None
        self.label = None
        self.cur = 0
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.samples)
	self.get_batch()   
	 
       
    def provide_data(self):
        return [(k, v.shape) for k, v in self.data.items()]   
    
    def provide_label(self):
        return [(k, v.shape) for k, v in self.label.items()]
    
    def reset(self):
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.samples)
            
    def iter_next(self):
        return self.cur < self.num_classes
        
    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.num_genuine
            return mx.io.DataBatch(data=self.data, label=self.label)
        else:
            raise StopIteration
 
    def get_batch(self):
        cur = self.cur
        num_classes = self.num_classes
        num_genuine = self.num_genuine
        num_imposter = self.num_imposter
        data_genuine_index = [self.samples[i%num_classes] for i in range(cur, cur + num_genuine)]
        data_genuine = []
        label_genuine = []
        label_genuine_set =[]
        
        for i in data_genuine_index:
            file = self.data_dir + "/" + i
            x = np.load(file)
            data_genuine.append(x)
            label_genuine.append(int(i)*np.ones(x.shape[0]))
            label_genuine_set += [int(i)]
            
            
        data_imposter_index = [self.samples[i%num_classes] for i in range( cur + num_genuine,  cur + num_classes)]
        data_imposter_index = [data_imposter_index[i] for i in np.random.randint(0, len(data_imposter_index), num_imposter)]
        data_imposter = []
        label_imposter = []
        for i in data_imposter_index:
            file = self.data_dir + "/" + i
            x = np.load(file)
            data_imposter.append(x)
            label_imposter.append(int(i)*np.ones(x.shape[0]))
                
        
        self.data = {'data_genuine': np.concatenate(data_genuine), 'data_imposter': np.concatenate(data_imposter)}
        self.label = {'label_genuine': np.concatenate(label_genuine), 'label_imposter': np.concatenate(label_imposter), 'label_genuine_set': np.array(label_genuine_set)}

