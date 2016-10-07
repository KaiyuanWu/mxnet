# -*- coding: utf-8 -*-
import numpy as np
import pairwise_distance_gpu as pdg
from scipy.sparse import coo_matrix
from collections import Counter
import time

class VerifyLoss:
    def __init__(self, margin = 1, mu = 0.5):
        self.margin = margin
        self.mu = mu
	self.debug = {}
        
    def get_mu(self):
        return self.mu
    
    def get_margin(self):
        return self.margin
    
    def approx_gradient(self, data, genuine_pair, imposter_triplet, strategy):
        grad = np.copy(data)
        epsilon = 1.e-6
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                data0 = np.copy(data)
                data0[i,j] = data[i,j] - 0.5*epsilon
                f0 = self.eval(data0, genuine_pair, imposter_triplet, strategy)
                data0[i,j] = data[i,j] + 0.5*epsilon
                f1 = self.eval(data0, genuine_pair, imposter_triplet, strategy)
                grad[i,j] = (f1 - f0)/epsilon
        return grad        
    """
    Genuine Pair: 
    0  None
    1  all k
    2  k-st
    3  all
    Imposter Triplet:
    0  None
    1  k-st
    2  all k
    """
    def forward_backward(self, data, label, genuine_label, strategy = (2,1), k=3, calculate_loss = True):
        if strategy[0] == 1 or strategy[0] == 2 or strategy[1] == 1 or strategy[1] == 2:
		self._select_knn(data, label, genuine_label, k) 
        genuine_pair, genuine_pair_dist = self._select_genuine_pair(label, genuine_label, strategy[0])
        imposter_triplet, imposter_triplet_dist = self._select_imposter_triplet(data, label, genuine_label, strategy[1])
        
        nsamples = data.shape[0]
        

        if strategy[0] != 0:
            AG = coo_matrix((genuine_pair[...,0]*0-1, (genuine_pair[...,0], genuine_pair[...,1])), shape=(nsamples, nsamples))
            AG_row_sum = AG.sum(axis = 0)
            AG_col_sum = AG.sum(axis = 1)
	    T = coo_matrix((-0.5*(AG_row_sum.getA1() + AG_col_sum.getA1()), (range(nsamples), range(nsamples))), shape=(nsamples, nsamples))
	    AG = AG + T 
            AG = AG + AG.T
        else:
            AG = coo_matrix(([], ([], [])), shape=(nsamples, nsamples))
            
        if strategy[1] != 0 and imposter_triplet.shape[0]>0:
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
            loss = self.eval2( data, genuine_pair, genuine_pair_dist,  imposter_triplet, imposter_triplet_dist, strategy)
        else:
            loss = 0.
        return loss, grad
    def _select_knn(self, data, label, genuine_label, k):
        self.k_inds = {}
        self.k_dists = {}
        for l in genuine_label:
            inds, = np.nonzero(label == l)
            X = data[inds,...].astype('float32')
	    dist = np.zeros((X.shape[0], X.shape[0]), dtype='float32')
	    pdg.pairwise_dist_gpu1(X, dist)
	    np.fill_diagonal(dist, np.inf)
	    n1 = np.argpartition(dist, min(k, inds.shape[0]-1))
	    dist = dist[[[i] for i in range(n1.shape[0])],n1[...,:min(k, inds.shape[0]-1)]]
	    n2 = np.argsort(dist)
	    nn = n1[[[i] for i in range(n1.shape[0])], n2]
            self.k_inds[l] = inds[nn]
            ix = [[i] for i in range(inds.shape[0])]
            self.k_dists[l]= dist[ix, n2]
        
    def _select_genuine_pair(self, label, genuine_label, strategy):        
        genuine_pair = []
	genuine_pair_dist = 0
	if strategy == 1:
            for l in genuine_label:
                inds, = np.nonzero(label == l)
                k_inds = self.k_inds[l]
                for i in range(len(inds)):
                    a = np.copy(k_inds[i,...])
                    a[:] = inds[i]
                    genuine_pair = genuine_pair + zip(a, k_inds[i,...]) 
		    genuine_pair_dist +=  self.k_dists[l][i,...].sum()
                    
        if strategy == 2:
            for l in genuine_label:
                inds, = np.nonzero(label == l)
                k_inds = self.k_inds[l]
                for i in range(len(inds)):
                    genuine_pair = genuine_pair + [(inds[i], k_inds[i,-1])]
		    genuine_pair_dist +=  self.k_dists[l][i,-1]
                  
        if strategy == 3:
            for l in genuine_label:
                inds, = np.nonzero(label == l)
                num = len(inds)
                a = np.array(range(num))
                for i in range(num-1):
                    a[:] = inds[i]
                    b = inds[i+1:]
                    genuine_pair = genuine_pair + zip(a,b)
                    
        return  np.array(genuine_pair), genuine_pair_dist
                    
    def _select_imposter_triplet(self, data, label, genuine_label, strategy):
        imposter_triplet = []
	imposter_triplet_dist = 0
        if strategy == 0:
            for l in genuine_label:
                margin = self.margin
                in_inds, = np.nonzero(label == l)
                out_inds, = np.nonzero(label != l)
                X = data[out_inds,...].astype('float32')
		Y = data[in_inds,...].astype('float32')
		dist = np.zeros((X.shape[0], Y.shape[0]), dtype='float32')
		pdg.pairwise_dist_gpu2(X,Y, dist)
		i,j = np.nonzero(dist < margin)
                imposter_triplet =  imposter_triplet + zip(in_inds[j], j, out_inds[i])
		imposter_triplet_dist += (margin - dist[i,j]).sum()
        if strategy == 1:
            for l in genuine_label:
                index_a = self.k_inds[l][...,-1]
                margin = self.k_dists[l][...,-1].flatten() + self.margin
                in_inds, = np.nonzero(label == l)
                out_inds, = np.nonzero(label != l)
                X = data[out_inds,...].astype('float32')
                Y = data[in_inds,...].astype('float32')
                dist = np.zeros((X.shape[0], Y.shape[0]), dtype='float32')
		pdg.pairwise_dist_gpu2(X,Y, dist)
		i,j = np.nonzero(dist < margin)
		imposter_triplet =  imposter_triplet + zip(in_inds[j], index_a[j], out_inds[i])
               	imposter_triplet_dist += (margin[j]-dist[i,j]).sum()
        if strategy == 2:
            raise Exception("This mode of impotser triplet has not been implemented!")            
        return np.array(imposter_triplet), imposter_triplet_dist
        
    
    def eval(self, data, genuine_pair, imposter_triplet, strategy):
        ret = 0
        s1 = 0
        s2 = 0
        for p in genuine_pair:
            diff = data[p[0],:] - data[p[1],:]
            s1 += (diff*diff).sum()
        ret +=  s1*0.5*self.mu
        
        for t in imposter_triplet:
            if strategy[1] != 0:
                diff1 = data[t[0],:] - data[t[1],:]
                diff2 = data[t[0],:] - data[t[2],:]
                s2 += (diff1*diff1).sum() + self.margin - (diff2*diff2).sum()
	    else:
                diff2 = data[t[0],:] - data[t[2],:]
                s2 += self.margin - (diff2*diff2).sum()
        ret += s2*0.5*(1-self.mu)
        return ret

    def eval2(self, data, genuine_pair, genuine_pair_dist, imposter_triplet, imposter_triplet_dist, strategy):
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
			if strategy[1] != 0:
				diff1 = data[t[0],:] - data[t[1],:]
				diff2 = data[t[0],:] - data[t[2],:]
				s2 += (diff1*diff1).sum() + self.margin - (diff2*diff2).sum()
			else:
				diff2 = data[t[0],:] - data[t[2],:]    
				s2 += self.margin - (diff2*diff2).sum()
	ret += s2*0.5*(1-self.mu)                                           
        return ret
