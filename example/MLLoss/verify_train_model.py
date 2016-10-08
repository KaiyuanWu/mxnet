# -*- coding: utf-8 -*-
#根据提供的的symbol，优化算法，数据文件夹data_dir，每个batch的num_imposter, num_genuine, 以及多少个epoch, sub_epoch
#以及优化算法来训练一个verify模型
import numpy as np
import mxnet  as mx
import verify_data_iter
import verify_loss
import logging

def VerifyTrain():
    def __init__(self):
        FORMAT = "%(asctime)-15s  %(message)s"
        logging.basicConfig(format=FORMAT)
        self.loss_func = verify_loss.LMNN()
        
    def train(self, symbol,  data_dir, num_imposter, num_genuine, img_size, epochs, sub_epochs, batch_size):
        logging.warning("initialize the data iteration")
        data_iter = verify_data_iter.MajorDataIter(data_dir, num_genuine = num_genuine, 
            num_imposter = num_imposter, img_size=img_size, shuffle=True)
        logging.warning("bind the symbol")
        
        input_variables_shape = {'data': (batch_size, img_size[0], img_size[1], img_size[2])}
        exe = symbol.simple_bind(ctx=mx.cpu(), **input_variabels_shape)
        
        param_idx2name = {}
        param_name2idx = {}
        idx = 0
        for param in exe.arg_dict():
            if param not in input_variabels_shape:
                param_idx2name[param] = idx 
                param_name2idx[idx] = param
                idx += 1
        print(param_idx2name, param_name2idx)
        
        logging.warning("init optimizer")
        optimizer = mx.optimizer.SGD(learning_rate = 0.001,
                            momentum = 0.9,
                            wd = 0.00001, param_idx2name=param_idx2name)
        state={}
        for grad in exe.grad_dict:
            if grad not in ['data']:
                state[grad] = optimizer.create_state(index =param_name2idx[grad], weight=exe.arg_dict[grad])  
        logging.warning("start to train ...")
        for epoch in range(epochs):
            for bacth in data_iter:
                data = batch.data['data']
                label = batch.label['label']
                genuine_label_set = batch.label['genuine_label_set']
                feat_buf = np.zeros((data.shape[0], exe.outputs[0].shape[1]), dtype='float32')
                index = range(0, data.shape[0], batch_size)
                for isub_epoch in range(sub_epchs):
                    for sub_batch in range(len(index)-1):
                        exe.arg_dict['data'][:] = data[index[sub_batch]:index[sub_batch + 1],...]
                        exe.forward()
                        feat_buf[index[sub_batch]:index[sub_batch + 1],...] = exe.outputs[0].asnumpy()
                    if index[-1] < sub_data.shape[0]:
                        temp_index = [i%sub_data.shape[0] for i in range(index[-1],index[-1]+batch_size)]
                        data[:] = sub_data[temp_index,...]
                        exe.forward()
                        feat_buf[index[-1]:sub_data.shape[0],...] = exe.outputs[0].asnumpy()[0:sub_data.shape[0]-index[-1],...]
                    logging.debug("epoch %d sub_epoch %d calculate loss"%(epoch, sub_epoch))     
                    loss, grad = loss_funcself.loss_func.forward_backward(feat_buf, label, genuine_label_set)
                    logging.debug("epoch %d sub_epoch %d backward"%(epoch, sub_epoch))
                    for sub_batch in range(len(index)-1):
                        data[:] = sub_data[index[sub_batch]:index[sub_batch + 1],...]
                        exe.forward()
                        output_grad[:] = grad[index[sub_batch]:index[sub_batch + 1],...]
                        exe.backward(output_grad, grad_req ='add')
                    if index[-1] < sub_data.shape[0]:
                        temp_grad[:] = 0
                        temp_grad[0:sub_data.shape[0]-index[-1],...] = grad[index[-1]:sub_data.shape[0],...]
                        temp_index = [i%sub_data.shape[0] for i in range(index[-1],index[-1]+sub_batch_size)]
                        data[:] = sub_data[temp_index,...]
                        exe.forward(is_train=True)
                        output_grad[:] = temp_grad
                        exe.backward(output_grad,grad_req ='add')
                    logging.debug("epoch %d sub_epoch %d update weights"%(epoch, sub_epoch))    
                    optimizer.rescale_grad = 1./sub_data.shape[0]  
                    for grad in exe.grad_dict:
                        if grad not in ['data']:
                            optimizer.update(index = param_name2idx[grad], weight=exe.arg_dict[grad], grad = exe.grad_dict[grad], state=state[grad])
                            exe.grad_dict[grad][:] = 0
            dataiter.reset()  
