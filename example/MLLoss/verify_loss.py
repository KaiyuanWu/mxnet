# -*- coding: utf-8 -*-
# 简化veryfy_loss的逻辑
# 只考虑k-nearest neightbor的genuine pair
#     误差最大的k个imposter triplet
import numpy as np
import pairwise_distance_gpu as pdg_gpu
from sklearn.metrics import pairwise_distances as pdg_cpu
from scipy.sparse import coo_matrix
from collections import Counter
import  verify_loss_c_py as vlc

class VerifyLoss:
    def __init__(self, nclasses, nfeats, margin=1):
        self.margin_ = margin
        self.nclasses_ = nclasses
        self.nfeats_ = nfeats
        # self.centers_ = np.zeros((nclasses, nfeats), dtype = 'float32')
        self.centers_ = np.random.randn(nclasses, nfeats).astype('float32')

    def approx_gradient(self, data, triplet):
        grad = np.copy(data)
        epsilon = 1.
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                data0 = np.copy(data)
                data0[i, j] = data[i, j] - 0.5 * epsilon
                f0 = self.eval2(data0, triplet)
                data0[i, j] = data[i, j] + 0.5 * epsilon
                f1 = self.eval2(data0, triplet)
                grad[i, j] = (f1 - f0) / epsilon
        return grad

    def forward_backward(self, data, label, k=3, center_update_lr=0., calculate_loss=True):
        unique_label, label = np.unique(label, return_inverse = True)
        label = label.astype('int32')
        center_data = self.centers_[unique_label]

        ext_data = np.concatenate((data, center_data), axis=0)
        #print(label.shape, unique_label.shape)
        ext_label = np.concatenate((label, np.arange(unique_label.shape[0],dtype='int32')), axis=0)
        num_data = data.shape[0]
        num_center_data = center_data.shape[0]
        num_ext_data = ext_data.shape[0]

        grad2 = ext_data.copy()
        loss2 = vlc.verify_loss_c_py(X=ext_data, label = ext_label, num_extdata = unique_label.shape[0],
                                          margin = self.margin_, k=k, grad = grad2)
        print('loss2 = ', loss2)
        print('grad2 = ', grad2)
        # CPU版本的pairwise distance计算
        # dist = pdg_cpu(ext_data, metric='sqeuclidean')
        # GPU版本的pairwise distance计算
        dist = np.zeros((num_ext_data, num_ext_data), dtype='float32')
        pdg_gpu.pairwise_dist_gpu1(ext_data, dist)

        inds = np.argpartition(dist[:num_data], k + 1)

        dist_to_center = [dist[i, num_data + label[i]] for i in range(num_data)]
        triplet = []
        for i in range(num_data):
            center_idx = num_data + label[i]
            label_i = label[i]
            t = []
            d = []
            l = []
            for j in range(k + 1):
                if dist[i, inds[i, j]] < dist_to_center[i] + self.margin_ and ext_label[inds[i, j]] != label_i:
                    triplet.append([i, center_idx, inds[i, j]])
                    t.append(inds[i,j])
                    d.append(dist[i, inds[i, j]])
            #print(i, 'label:', label_i, t, dist[i, inds[i,:k+1]], ext_label[inds[i,:k+1]])

        A = np.zeros((num_ext_data, num_ext_data))
        C = np.array([[0.75, -1.25, 0.5],
                      [-1.25, 0.75, 0.5],
                      [0.5, 0.5, -1.]])
        for idx in triplet:
            A[[[idx[0]], [idx[1]], [idx[2]]], idx] += C
        """
        print("A = ")
        for i in range(num_ext_data):
            print([x for x in A[i]])
        """
        grad = np.dot(A, ext_data)

        """
        print("grad = ")
        for i in range(num_ext_data):
            print([x for x in grad[i]])
        """
        if center_update_lr > 0:
            self.centers_[unique_label] -= center_update_lr / num_data * grad[num_data:]
        #print(self.centers_)

        if calculate_loss:
            """
            p = ext_data.reshape(-1)
            q = grad.reshape(-1)
            print('-------------------------------')
            for i in range(p.shape[0]):
                print("%f*%f"%(p[i], q[i]))
            """
            loss = 0.5 * sum(sum(ext_data * grad)) #self.eval(ext_data, A)
        else:
            loss = 0.
        # grad2 = self.approx_gradient(ext_data, triplet)
        # print('approx grad = ', grad2, 'calculate grad', grad, ' diff ', np.linalg.norm(grad-grad2))
        return loss, grad[:num_data], len(triplet)

    def eval(self, data, A):
        ret = 0.5 * sum(sum(data * (np.dot(A, data))))
        return ret

    def eval2(self, data, triplet):
        ret = 0.
        for t in triplet:
            x = data[t[0]]
            y = data[t[1]]
            z = data[t[2]]
            ret += sum((x - y) * (x - y)) + self.margin_ - sum((z - 0.5 * (x + y)) * (z - 0.5 * (x + y)))
        return 0.5 * ret
