import os
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "2"
import mxnet as mx
import numpy as np

import scipy.io as sio

class GLMatcher(mx.operator.CustomOp):
    def __init__(self):
        self.init = False

    def forward(self,is_train, req, in_data, out_data, aux):
        global_pred = in_data[0].asnumpy()
        local_pred = in_data[1].asnumpy()

        if self.init == False:
            _, _, self.grid_width, self.grid_height = local_pred.shape
            #print(local_pred.shape)
            #quit()
            self.num_images, self.num_predicts = global_pred.shape
            #Each prediction corresponds to a point's coordinate, i.e., (x,y)
            self.num_predicts /= 2
            #keep the match index
            self. match = np.zeros((self.num_images, 2*self.num_predicts), dtype = 'int32')
            self.local_grad = in_data[1].asnumpy()
            self.init = True
            
        self.find_match(global_pred)
        for img in range(self.num_images):
            for ip in range(self.num_predicts):
                global_pred[img, ip] += local_pred[img, ip,
                                            self.match[img,ip], self.match[img, self.num_predicts + ip]]
                global_pred[img, self.num_predicts + ip] += local_pred[img, self.num_predicts + ip,
                                            self.match[img,ip], self.match[img, self.num_predicts + ip]]
                
        self.assign(out_data[0], req[0], mx.nd.array(global_pred))

    def find_match(self, global_pred):
        for img in range(self.num_images):
            for ip in range(self.num_predicts):
                if global_pred[img, ip]<= 0:
                    self.match[img, ip] = 0
                else:
                    if global_pred[img, ip] >= 1:
                        self.match[img,ip] = self.grid_width - 1
                    else:
                        self.match[img,ip] = int(global_pred[img, ip]*self.grid_width)
                if global_pred[img, self.num_predicts + ip] <= 0:
                    self.match[img, self.num_predicts + ip] = 0
                else:
                    if global_pred[img, self.num_predicts + ip] >= 1:
                        self.match[img,self.num_predicts + ip] = self.grid_height - 1
                    else:
                        self.match[img,self.num_predicts + ip] = int(global_pred[img, self.num_predicts + ip]*self.grid_height)

        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_data[0].asnumpy() - in_data[2].asnumpy()
        self.local_grad[:] = 0.
        for img in range(self.num_images):
            for ip in range(self.num_predicts):
                self.local_grad[img, ip, self.match[img,ip], self.match[img, self.num_predicts + ip]] = y[img, ip]
                self.local_grad[img, self.num_predicts + ip, self.match[img,ip], self.match[img, self.num_predicts + ip]] = y[img, self.num_predicts + ip]
        
        self.assign(in_grad[0], req[0], mx.nd.array(y))
        self.assign(in_grad[1], req[1], mx.nd.array(self.local_grad))
@mx.operator.register("glmatcher")
class GLMatcherProp(mx.operator.CustomOpProp):
     def __init__(self):
         super(GLMatcherProp, self).__init__(need_top_grad = False)

     def list_arguments(self):
        return['global_pred','local_pred','label']

     def list_outputs(self):
         return['prediction']

     def infer_shape(self, in_shape):
         output_shape = in_shape[0]
         return in_shape, [output_shape], []  

     def create_operator(self, ctx, shapes, dtypes):
         return GLMatcher()
