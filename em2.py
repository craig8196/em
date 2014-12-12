from __future__ import division, print_function

import os
import re
import sys
import math
import numpy as np
from scipy.misc import logsumexp
from scipy.special import gammaln
import codecs
import random

# for 20 newsgroups this script assumes the groups directory
# is at the same level as this directory


class EM(object):
    def __init__(self):
        self.log_likelihoods = []
    
    # Not a mixture of multinomials.
    def _check_threshold(self):
        self.curr_iter += 1
        if self.curr_iter > self.max_iter:
            return False
        self.x_log_beta = self.log_betas.dot(self.processed_data.T)
        self.likelihoods = np.array([logsumexp(row+self.log_lambdas) for row in self.x_log_beta.T])
        #~ print(self.likelihoods.sum())
        temp = self.x_log_beta.copy()
        shape = temp.shape
        for k in xrange(shape[0]):
            for i in xrange(shape[1]):
                temp[k][i] += -self.log_fact_processed_data_per_doc[i] + self.log_fact_data_token_counts[i] + self.log_lambdas[k]
        likelihood_of_data = np.array([logsumexp(row) for row in temp.T]).sum()
        self.log_likelihoods.append(likelihood_of_data)
        previous = self.previous_likelihood
        self.previous_likelihood = likelihood_of_data
        
        print("Current Likelihood of Data:", self.previous_likelihood)
        if abs(previous - likelihood_of_data) < self.likelihood_threshold:
            return False
        return True # Keep iterating.
    
    def run_em(self, clusters, data, likelihood_threshold=10**-2, max_iter=200):
        """Performs EM to classify the given data.
        Does not perform feature selection for you.
        """
        # set thresholds
        self.likelihood_threshold = likelihood_threshold
        self.previous_likelihood = float("-inf")
        self.max_iter = max_iter
        self.curr_iter = 0
        
        # transform to counts and get vocabulary
        self.clusters = clusters
        self.doc_count = len(data)
        processed_data = []
        # determine word types
        types = set()
        for doc in data:
            for tok in doc:
                types.add(tok)
        vocab_map = {}
        vocab = []
        # map words to indices and vice versa
        for index, t in enumerate(types):
            vocab.append(t)
            vocab_map[t] = index
        processed_data = np.zeros((len(data), len(types)))
        for i, doc in enumerate(data):
            for tok in doc:
                j = vocab_map[tok]
                processed_data[i, j] += 1
        self.processed_data = processed_data
        self.log_fact_data_token_counts = np.array([gammaln(row.sum()+1) for row in self.processed_data])
        self.log_fact_processed_data_per_doc = np.array([row.sum() for row in gammaln(processed_data + np.ones(self.processed_data.shape))])
        self.log_fact_multinomial_coefficients = self.log_fact_data_token_counts - self.log_fact_processed_data_per_doc
        #~ print(self.log_fact_multinomial_coefficients) # for toy dataset should sum to one
        self.vocab = vocab
        self.vocab_map = vocab_map
        # get initial lambdas
        #~ self.log_lambdas = np.log(np.array([0.3, 0.7])) # used for testing
        self.log_lambdas = np.log(np.array([1/clusters for i in xrange(clusters)]))
        # get initial betas
        #~ self.log_betas = np.log(np.array([np.array([0.3, 0.7]), np.array([0.6, 0.4])])) # used for testing
        self.log_betas = np.log(np.random.dirichlet([0.1 for i in xrange(len(types))], clusters))
        
        # conduct EM
        while self._check_threshold(): # check threshold performs initial data crunching
            # calc new lambdas
            new_lambdas = []
            log_doc_count = np.log(self.doc_count)
            for k in xrange(clusters):
                tmp = np.array([self.log_lambdas[k] for i in xrange(self.doc_count)])
                new_l = logsumexp(self.x_log_beta[k]+tmp-self.likelihoods) - log_doc_count
                new_lambdas.append(new_l)
            
            new_betas = np.copy(self.log_betas)
            t_proc_data = np.log(self.processed_data.T)
            for k in xrange(clusters):
                tmp = np.array([self.log_lambdas[k] for i in xrange(self.doc_count)])
                logprob_cluster_given_doc = self.x_log_beta[k] - self.likelihoods + tmp
                for t in xrange(len(self.vocab)):
                    new_betas[k, t] = logsumexp(t_proc_data[t]+logprob_cluster_given_doc)
                temp = logsumexp(new_betas[k])
                temp = logsumexp([temp, np.log(len(vocab))])
                for t in xrange(len(self.vocab)):
                    new_betas[k, t] = logsumexp([new_betas[k, t], 0]) - temp
            
            #~ for i in xrange(clusters): # make sure betas don't violate laws of probability
                #~ print(np.exp(new_betas[i]).sum())
            
            self.log_lambdas = new_lambdas
            self.log_betas = new_betas
        return self.cluster()
        
    def cluster(self):
        # make cluster assignments to document
        data = self.processed_data
        
        results = []
        for doc in data:
            max_cluster = 0
            max_prob = float('-inf')
            for k in xrange(self.clusters):
                logprob = self.log_betas[k].dot(doc)
                logprob += self.log_lambdas[k]
                if logprob > max_prob:
                    max_prob = logprob
                    max_cluster = k
            results.append(max_cluster)
        return results

