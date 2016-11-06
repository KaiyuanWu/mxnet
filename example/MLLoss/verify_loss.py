# -*- coding: utf-8 -*-
# 简化veryfy_loss的逻辑
import numpy as np
import mxnet as mx
import pairwise_distance_gpu as pdg


class VerifyLoss:
    def __init__(self, num_features, margin = 1):
        self.margin_ = margin
        self.num_features_ = num_features
        x = mx.symbol.Variable('x')
        nx = mx.symbol.L2Normalization(data = x, name = "x_norm")
        self.x_data = mx.nd.zeros((1,num_features), ctx = mx.cpu())
        self.x_grad = mx.nd.zeros((1,num_features), ctx = mx.cpu())
        self.exe = nx.bind(ctx = mx.cpu(), args={'x':self.x_data}, args_grad = {'x':self.x_grad}, grad_req = 'write')
        self.x_norm = self.exe.outputs[0]
        self.nx_grad = self.x_norm.copy()


    def forward_backward(self, data, label, center_data):
        unique_label, new_label = np.unique(label, return_inverse = True)
        centers = center_data[unique_label]
        num_data = data.shape[0]
        num_centers = centers.shape[0]
        centers_norm = centers.copy()
        # Norm centers
        for i in range(num_centers):
            #print(centers[i])
            self.x_data[0] = centers[i]
            self.exe.forward(is_train= True)
            centers_norm[i] = self.x_norm.asnumpy()
            #print(centers_norm[i])
        #quit()

        dist = np.zeros((num_data, num_centers), dtype = 'float32')
        pdg.pairwise_dist_gpu2(data, centers_norm, dist)

        loss = 0.
        num_pos_pair = 0
        num_neg_pair = 0
        A = np.zeros((num_data + num_centers, num_data + num_centers))
        for idx in range(num_data):
            #max_neg_loss = 0
            #max_neg_c = -1
            for c in range(num_centers):
                if c != label[idx]:
                    if  1.*self.margin_ > dist[idx, c] :
                        #max_neg_loss = 1.*self.margin_ - dist[idx, c]
                        #max_neg_c = c
                        A[idx, idx] -= 1.
                        A[idx, num_data + c] += 1
                        A[num_data + c, idx] += 1
                        A[num_data + c, num_data + c] -= 1
                        loss += 1.*self.margin_ - dist[idx, c]
                        num_neg_pair += 1
                else:
                    if dist[idx, c] > .5*self.margin_:
                        A[idx, idx] += 1
                        A[idx, num_data + c] -= 1
                        A[num_data + c, idx] -= 1
                        A[num_data + c, num_data + c] += 1
                        loss += dist[idx, c] - .5 * self.margin_
                        num_pos_pair += 1
            """
            if max_neg_c != -1:
                A[idx, idx] -= 1.
                A[idx, num_data + max_neg_c] += 1
                A[num_data + max_neg_c, idx] += 1
                A[num_data + max_neg_c, num_data + max_neg_c] -= 1
                loss += max_neg_loss
                num_neg_pair += 1
            """

        ext_data = np.concatenate((data, centers), axis=0)
        grad = 1./num_centers*np.dot(A, ext_data)
        #print('ext_data = ', ext_data)
        #print('grad = ', grad)
        loss /= num_data
        center_grad = np.zeros(center_data.shape)
        # Backpropogate the gradient
        for i in range(num_centers):
            self.x_data[0] = centers[i]
            self.exe.forward(is_train=True)
            self.nx_grad[0] = grad[num_data+i]
            #print(centers[i], grad[num_data+i])
            self.exe.backward(self.nx_grad)
            grad[num_data + i] = self.x_grad.asnumpy()
            #print(grad[num_data + i])
        center_grad[unique_label] = grad[num_data:]
        return loss, grad[:num_data], center_grad, num_pos_pair, num_neg_pair
