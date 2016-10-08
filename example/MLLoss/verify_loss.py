# -*- coding: utf-8 -*-
#简化veryfy_loss的逻辑
#只考虑k-nearest neightbor的genuine pair
#     误差最大的k个imposter triplet
import numpy as np
#import pairwise_distance_gpu as pdg
from sklearn.metrics import pairwise_distances as pdg
from scipy.sparse import coo_matrix
from collections import Counter

class VerifyLoss:
    def __init__(self, margin = 1, mu = 0.5):
        self.margin = margin
        self.mu = mu
        
    def get_mu(self):
        return self.mu
    
    def get_margin(self):
        return self.margin
    
    def approx_gradient(self, data, genuine_pair, imposter_triplet):
        grad = np.copy(data)
        epsilon = 1.
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                data0 = np.copy(data)
                data0[i,j] = data[i,j] - 0.5*epsilon
                f0 = self.eval(data0, genuine_pair, 0, imposter_triplet,0)
                data0[i,j] = data[i,j] + 0.5*epsilon
                f1 = self.eval(data0, genuine_pair, 0, imposter_triplet, 0)
                grad[i,j] = (f1 - f0)/epsilon
        return grad        
    
    def forward_backward(self, data, label, genuine_label,  k=3, calculate_loss = True):
        self._select_knn(data, label, genuine_label, k)
        
        genuine_pair, genuine_pair_dist = self._select_genuine_pair(label, genuine_label)
        imposter_triplet, imposter_triplet_dist = self._select_imposter_triplet(data, label, genuine_label)
        
        nsamples = data.shape[0]
        
        AG = coo_matrix((genuine_pair[...,0]*0-1, (genuine_pair[...,0], genuine_pair[...,1])), shape=(nsamples, nsamples))
        AG_row_sum = AG.sum(axis = 0)
        AG_col_sum = AG.sum(axis = 1)
	T = coo_matrix((-0.5*(AG_row_sum.getA1() + AG_col_sum.getA1()), (range(nsamples), range(nsamples))), shape=(nsamples, nsamples))
	AG = AG + T 
        AG = AG + AG.T
        
        if imposter_triplet.shape[0]>0:
            BIs = Counter(imposter_triplet[...,0]*nsamples + imposter_triplet[...,1])
            k = np.array(BIs.keys(), dtype='int')
            v = np.array(BIs.values())*(-1)
            i = k%nsamples
            j = k/nsamples
            BI = coo_matrix((v, (i,j)), shape=(nsamples, nsamples))
            BI_row_sum = BI.sum(axis = 0)
            BI_col_sum = BI.sum(axis = 1)
	    T = coo_matrix((-0.5*(BI_row_sum.getA1() + BI_col_sum.getA1()), (range(nsamples), range(nsamples))), shape=(nsamples, nsamples))
            BI = BI + T
            BI = BI + BI.T
        else:
            BI = coo_matrix(([], ([], [])), shape=(nsamples, nsamples))
        
        
        if imposter_triplet.shape[0]>0:
            CIs = Counter(imposter_triplet[...,0]*nsamples + imposter_triplet[...,2])
            k = np.array(CIs.keys(), dtype='int')
            v = np.array(CIs.values())*(-1)
            i = k%nsamples
            j = k/nsamples
            CI = coo_matrix((v, (i,j)), shape=(nsamples, nsamples))
            CI_row_sum = CI.sum(axis = 0)
            CI_col_sum = CI.sum(axis = 1)
            #CI.setdiag(-0.5*(CI_row_sum.getA1() + CI_col_sum.getA1()))
	    T = coo_matrix((-0.5*(CI_row_sum.getA1() + CI_col_sum.getA1()), (range(nsamples), range(nsamples))), shape=(nsamples, nsamples))
            CI = CI + T
            CI = CI + CI.T
        else:
            CI = coo_matrix(([], ([], [])), shape=(nsamples, nsamples))

        grad = (self.mu*AG + (1-self.mu)*(BI-CI))*data
        
        if calculate_loss:
            loss = self.eval(data, genuine_pair, genuine_pair_dist,  imposter_triplet, imposter_triplet_dist)
        else:
            loss = 0.
        #grad2 = self.approx_gradient(data, genuine_pair, imposter_triplet)
        #print('approx grad = ', grad2)
        return loss, grad
    def _select_knn(self, data, label, genuine_label, k):
        self.k = k
        self.k_inds = {}
        self.k_dists = {}
        for l in genuine_label:
            inds, = np.nonzero(label == l)
            #CPU版本的pairwise distance计算
            dist = pdg(data[inds,...], metric='sqeuclidean')
            #GPU版本的pairwise distance计算
            #X = data[inds,...].astype('float32')
	    #dist = np.zeros((X.shape[0], X.shape[0]), dtype='float32')
	    #pdg.pairwise_dist_gpu1(X, dist)
	    np.fill_diagonal(dist, np.inf)
	    n1 = np.argpartition(dist, min(k, inds.shape[0]-1))
	    dist = dist[[[i] for i in range(n1.shape[0])],n1[...,:min(k, inds.shape[0]-1)]]
	    n2 = np.argsort(dist)
	    nn = n1[[[i] for i in range(n1.shape[0])], n2]
            self.k_inds[l] = inds[nn]
            self.k_dists[l]= dist[[[i] for i in range(inds.shape[0])], n2]
    def _select_genuine_pair(self, label, genuine_label):        
        genuine_pair = []
	genuine_pair_dist = 0
        for l in genuine_label:
            inds, = np.nonzero(label == l)
            k_inds = self.k_inds[l]
            for i in range(len(inds)):
                a = np.copy(k_inds[i,...])
                a[:] = inds[i]
                genuine_pair = genuine_pair + zip(a, k_inds[i,...]) 
		genuine_pair_dist +=  self.k_dists[l][i,...].sum()
                    
        return  np.array(genuine_pair), genuine_pair_dist
                    
    def _select_imposter_triplet(self, data, label, genuine_label):
        imposter_triplet = []
	imposter_triplet_dist = 0
        for l in genuine_label:
            index_a = self.k_inds[l][...,-1]
            margin = self.k_dists[l][...,-1].flatten() + self.margin
            in_inds, = np.nonzero(label == l)
            out_inds, = np.nonzero(label != l)
            #CPU版本的pairwise distance计算
            dist = pdg(data[out_inds,...], data[in_inds,...],metric='sqeuclidean')
            #GPU版本的pairwise distance计算
            #X = data[out_inds,...].astype('float32')
            #Y = data[in_inds,...].astype('float32')
            #dist = np.zeros((X.shape[0], Y.shape[0]), dtype='float32')
            #pdg.pairwise_dist_gpu2(X,Y, dist)
            i,j = np.nonzero(dist >= margin)
            dist[i,j] = np.inf
            j = np.argpartition(dist, self.k, )[...,0:self.k].flatten()
            i = np.tile(range(dist.shape[0]),[self.k,1]).T.flatten()
            for p in zip(i,j):
                if dist[p[0],p[1]] < np.inf:
                        imposter_triplet.append([in_inds[p[1]], index_a[p[1]], out_inds[p[0]]])
                        imposter_triplet_dist += (margin[p[1]]-dist[p[0],p[1]])        
        return np.array(imposter_triplet), imposter_triplet_dist
        
    
    def eval(self, data, genuine_pair, genuine_pair_dist, imposter_triplet, imposter_triplet_dist):
	ret = 0
        s1 = 0
        s2 = 0
	if genuine_pair_dist != 0:
            s1 = genuine_pair_dist
	else:
            s1 = 0
            for p in genuine_pair:
		diff = data[p[0],:] - data[p[1],:]
		s1 += (diff*diff).sum()   
	ret +=  s1*0.5*self.mu
	if imposter_triplet_dist != 0:
            s2 = imposter_triplet_dist
        else:
            s2 = 0
            for t in imposter_triplet:
                diff1 = data[t[0],:] - data[t[1],:]
                diff2 = data[t[0],:] - data[t[2],:]
                s2 += (diff1*diff1).sum() + self.margin - (diff2*diff2).sum() 
	ret += s2*0.5*(1-self.mu)                                           
        return ret
