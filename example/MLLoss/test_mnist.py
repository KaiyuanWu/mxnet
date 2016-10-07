import numpy as np
import sys
import mxnet as mx
import cv2
import os
import sys
import cPickle
import scipy.io as sio
import verify_loss
import verify_data_iter

def load_data():
    #matfile = '/Users/kaiwu/Documents/veryes/Distance_Metric_Learning/experiments/mnist/mnist.mat'
    matfile = '/home/kaiwu/mxnet/experiments/mnist/mnist.mat'
    mat = sio.loadmat(matfile)
    train_x = mat['train_x']
    train_y = mat['train_y']
    test_x = mat['test_x']
    test_y = mat['test_y']
    return train_x, train_y, test_x, test_y
    
def save_data(x, y, prefix):
    nfeats,nx = x.shape
    data = {}
    for ix in range(nx):
        if y[ix][0] in data:
            data[y[ix][0]].append(x[...,ix])
        else:
            data[y[ix][0]] = [x[...,ix]]
    for label in data:
        fo = open("%s/%d"%(prefix, label), 'wb')
        cPickle.dump(data[label][0:100],fo)
        fo.close()
        
def preprocess_data():
    train_x, train_y, test_x, test_y = load_data()
    prefix = '/home/kaiwu/mxnet/experiments/mnist/'    
    save_data(train_x, train_y, prefix + 'train2/')
        
    
def get_lenet_symbol():
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data = data, kernel = (5, 5), num_filter = 20)
    relu1 = mx.symbol.Activation(data = conv1, act_type = "relu")
    pool1 = mx.symbol.Pooling(data = relu1, pool_type = "max", kernel = (2, 2), stride = (2, 2))
    # second conv
    conv2 = mx.symbol.Convolution(data = pool1, kernel = (5, 5), num_filter = 50)
    relu2 = mx.symbol.Activation(data = conv2, act_type = "relu")
    pool2 = mx.symbol.Pooling(data = relu2, pool_type = "max", kernel = (2, 2), stride = (2, 2))
    # first fullc
    flatten = mx.symbol.Flatten(data = pool2)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
    relu3 = mx.symbol.Activation(data = fc1, act_type = "relu")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data = relu3, num_hidden = 64)
    return fc2

def train_model():
    num_epoch = 1000
    batch_size = 3
    num_sub_epoch = 1
    sub_batch_size = 128
    
    data_dir = '/tmp/RAM/train/'
    data_iter = verify_data_iter.MajorDataIter(data_dir = data_dir, batch_size=batch_size, num_imposter=1000, img_size = (1,28,28), shuffle = True)
    feat_sym = get_lenet_symbol()
    
    
    input_shapes = {'data':(sub_batch_size, 1, 28,28)}
    exe = feat_sym.simple_bind(ctx=mx.gpu(0), **input_shapes)
    arg_arrays = exe.arg_dict
    data = arg_arrays['data']
    init = mx.init.Uniform(scale = 0.025)
    for name, arr in arg_arrays.items():
        if name not in input_shapes:
            init(name, arr)
   

    loss_func = verify_loss.LMNN()
    
    state = {}
    for i, grad in enumerate(exe.grad_arrays):
            if feat_sym.list_arguments()[i] == 'data':
                continue
            state[i] = grad.copy()
    output_grad = exe.outputs[0].copy()            
    temp_grad = exe.outputs[0].asnumpy()
    for epoch in range(num_epoch):
        data_iter.reset()
        batch = data_iter.next()
        sub_data = np.concatenate((batch.data['data_genuine'], batch.data['data_imposter']))
        sub_data = sub_data.reshape((sub_data.shape[0],1,28,28))
        sub_label = np.concatenate((batch.label['softmax_genuine_label'], batch.label['softmax_imposter_label']))
        genuine_label = batch.label['genuine_label']

	
	feat_buf = np.zeros((sub_data.shape[0], exe.outputs[0].shape[1]), dtype='float32')
        index = range(0, sub_data.shape[0], sub_batch_size)
        for sub_epoch in range(num_sub_epoch):
            for sub_batch in range(len(index)-1):
                data[:] = sub_data[index[sub_batch]:index[sub_batch + 1],...]
                exe.forward()
                feat_buf[index[sub_batch]:index[sub_batch + 1],...] = exe.outputs[0].asnumpy()
	    if index[-1] < sub_data.shape[0]:
		temp_index = [i%sub_data.shape[0] for i in range(index[-1],index[-1]+sub_batch_size)]
		data[:] = sub_data[temp_index,...]
                exe.forward()
		feat_buf[index[-1]:sub_data.shape[0],...] = exe.outputs[0].asnumpy()[0:sub_data.shape[0]-index[-1],...]
             
            loss, grad = loss_func.forward_backward(feat_buf, sub_label, genuine_label)
            print("epoch ",epoch, " subepoch ", sub_epoch," loss ", loss, "grad shape", grad.shape, " feat shape", feat_buf.shape)
            
            for s in state:
                state[s] *= 0
            
            for sub_batch in range(len(index)-1):
                data[:] = sub_data[index[sub_batch]:index[sub_batch + 1],...]
                exe.forward()
                output_grad[:] = grad[index[sub_batch]:index[sub_batch + 1],...]
                exe.backward(output_grad)
                for i, g in enumerate(exe.grad_arrays):
                    if feat_sym.list_arguments()[i] == 'data':
                        continue
                    state[i] += g
	    if index[-1] < sub_data.shape[0]:
		temp_grad[:] = 0
		temp_grad[0:sub_data.shape[0]-index[-1],...] = grad[index[-1]:sub_data.shape[0],...]
		temp_index = [i%sub_data.shape[0] for i in range(index[-1],index[-1]+sub_batch_size)]
		data[:] = sub_data[temp_index,...]
		exe.forward()
		output_grad[:] = temp_grad
		exe.backward(output_grad)
                for i, g in enumerate(exe.grad_arrays):
                    if feat_sym.list_arguments()[i] == 'data':
                        continue
                    state[i] += g
            for i, w in enumerate(exe.arg_arrays):
                if feat_sym.list_arguments()[i] == 'data':
                    continue
		w -= 0.000001/len(index)*state[i]
        mx.model.save_checkpoint('models/mnist', epoch, feat_sym, exe.arg_dict, exe.aux_dict)

train_model()
