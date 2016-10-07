# -*- coding: utf-8 -*-
import mxnet as mx
import random
import os
import cPickle
import numpy as np

class MajorDataIter(mx.io.DataIter):
    def __init__(self, data_dir, batch_size = 1, num_imposter = 100, img_size=(3, 32, 32), shuffle=False):
        super(MajorDataIter, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_imposter = num_imposter
        self.img_size = img_size
        
        self.all_data = {}
	self.get_all_data()
        self.data_stat = self.analyze_data()
        self.data = None
        self.cur = 0
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.samples)
	self.get_batch()   
	 
    def get_all_data(self):
	data_dir = self.data_dir
        samples = os.listdir(data_dir)
        for s in samples:
            file = data_dir + "/" + s
            data = self.unpickle_(file)
	    self.all_data[s] = data
       
    def provide_data(self):
        return [(k, v.shape) for k, v in self.data.items()]   
    
    def provide_label(self):
        return [(k, v.shape) for k, v in self.label.items()]
    
    def reset(self):
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.samples)
            
    def iter_next(self):
        return self.cur < self.data_stat['num_classes']
        
    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label)
        else:
            raise StopIteration
    def unpickle(self, file):   
	id = file.split('/')[-1]
	return self.all_data[id]
 
    def get_batch(self):
        batch_size = self.batch_size
        cur = self.cur
        num_classes = self.data_stat['num_classes']
        data_genuine_index = [ self.samples[i%num_classes] for i in range(cur, cur+batch_size)]
        data_genuine = []
        label_genuine = []
        label_genuine_set =[]
        
        for i in data_genuine_index:
            file = self.data_dir + "/" + i
            tmp = self.unpickle(file)
	    ntemp = len(tmp)
	    index = np.random.randint(0,ntemp, 2000)
            data_genuine += [tmp[i] for i in index]
            label_genuine += [int(i)]*len(index)
            label_genuine_set += [int(i)]
        data_imposter_index = [self.samples[i%num_classes] for i in range( cur + batch_size,  cur + num_classes)]
        index_imposter = {}
        for i in range(self.num_imposter):
            r = random.randint(0,len(data_imposter_index)-1)
            s = random.randint(0, self.data_stat['info'][data_imposter_index[r]]-1)
            if r in index_imposter:
                index_imposter[r].append(s)
            else:
                index_imposter[r] = [s]
                
        data_imposter = []
        label_imposter = []
        for i in index_imposter:
            file = self.data_dir + "/" + data_imposter_index[i]
            tmp = self.unpickle(file)
            data_imposter += [tmp[j] for j in index_imposter[i]]
            label_imposter += [int(data_imposter_index[i])]*len(index_imposter[i])
        
        self.data = {'data_genuine': np.array(data_genuine), 'data_imposter': np.array(data_imposter)}
        self.label = {'softmax_genuine_label': np.array(label_genuine), 'softmax_imposter_label': np.array(label_imposter), 'genuine_label': np.array(label_genuine_set)}
        
    
    def analyze_data(self):
        data_dir = self.data_dir
        samples = os.listdir(data_dir)
        data_stat = {}
        data_stat['num_classes'] = len(samples)
        total_samples = 0
        info = {}
        self.samples = samples
        for s in samples:
            file = data_dir + "/" + s
            data = self.unpickle(file)
            total_samples += len(data)
            info[s] = len(data)
        data_stat['info'] = info
        data_stat['total_samples'] = total_samples
        return data_stat
