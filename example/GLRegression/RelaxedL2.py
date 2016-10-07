import os
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class RelaxedL2(mx.operator.CustomOp):
    def __init__(self, weight = 1, threshold = 0):
        super(RelaxedL2, self).__init__()
        self.weight = weight
        self.threshold = threshold
        
    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = in_data[0].asnumpy() - in_data[1].asnumpy()
        index1 = x>self.threshold
        index2 = x<-self.threshold
        index3 = (x>-self.threshold)*(x<self.threshold)
        y = in_grad[0].asnumpy()
        y[index1] = 2*self.weight*(x[index1] - self.threshold)
        y[index2] = 2*self.weight*(x[index2] + self.threshold)
        y[index3] = 0
        self.assign(in_grad[0], req[0], mx.nd.array(y))
    
@mx.operator.register("relaxedl2")
class RelaxedL2Prop(mx.operator.CustomOpProp):
    def __init__(self, weight=1., threshold = 0):
        super(RelaxedL2Prop, self).__init__(need_top_grad = False)
        self.weight = weight
        self.threshold = threshold        
        
    def list_arguments(self):
        return ['predict','label']
    
    def list_outputs(self):
        return ['output']
    
    def infer_shape(self, in_shape):
        predict_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return [predict_shape, label_shape], [output_shape],[]
    
    def create_operator(self, ctx, shapes, dtypes):
        return RelaxedL2(self.weight, self.threshold)
