from __future__ import division, print_function

import os
import re
import sys
import math
import numpy as np
from scipy.misc import logsumexp
import codecs
import random


class EM(object):
    def __init__(self, likelihood_threshold=10**-2):
        self.threshold = likelihood_threshold
        self.previous_likelihood = float("-inf")
    
    def _log_likelihood_of_data(self):
        return self.likelihoods.sum()
    
    def _check_threshold(self):
        self.beta_dot_data = self.betas.dot(self.processed_data.T)
        self.likelihoods = np.array([logsumexp(row+self.lambdas) for row in self.beta_dot_data.T])
        previous = self.previous_likelihood
        self.previous_likelihood = self._log_likelihood_of_data()
        print("Current Likelihood of Data:", self.previous_likelihood)
        if abs(previous - self.previous_likelihood) < self.threshold:
            return False
        return True # Keep iterating.
    
    def init(self, clusters, data):
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
        self.vocab = vocab
        self.vocab_map = vocab_map
        # get initial lambdas
        #~ self.lambdas = np.array([0.3, 0.7])
        self.lambdas = np.array([1/clusters for i in xrange(clusters)])
        self.lambdas = np.log(self.lambdas)
        # get initial betas
        #~ self.betas = np.array([np.array([0.3, 0.7]), np.array([0.6, 0.4])])
        self.betas = np.random.dirichlet([0.1 for i in xrange(len(types))], clusters)
        self.betas = np.log(self.betas)
        
        # conduct EM
        while self._check_threshold():
            # calc new lambdas
            new_lambdas = []
            log_doc_count = np.log(self.doc_count)
            for k in xrange(clusters):
                tmp = np.array([self.lambdas[k] for i in xrange(self.doc_count)])
                new_l = logsumexp(self.beta_dot_data[k]+tmp-self.likelihoods) - log_doc_count
                new_lambdas.append(new_l)
            
            new_betas = np.copy(self.betas)
            t_proc_data = np.log(self.processed_data.T)
            for k in xrange(clusters):
                tmp = np.array([self.lambdas[k] for i in xrange(self.doc_count)])
                logprob_cluster_given_doc = self.beta_dot_data[k] - self.likelihoods + tmp
                for t in xrange(len(self.vocab)):
                    new_betas[k, t] = logsumexp(t_proc_data[t]+logprob_cluster_given_doc)
                temp = logsumexp(new_betas[k])
                temp = logsumexp([temp, np.log(len(vocab))])
                for t in xrange(len(self.vocab)):
                    new_betas[k, t] = logsumexp([new_betas[k, t], 0]) - temp
            
            self.lambdas = new_lambdas
            self.betas = new_betas
        
    def cluster(self, data=None):
        is_processed = False
        if data is None:
            is_processed = True
            data = self.processed_data
        
        if not is_processed:
            pass # TODO process data
        
        for doc in data:
            pass
        return None

def main2():
    print('Welcome')
    em = EM()
    
    test_data = [
        ('H', ['H', 'H', 'H']),
        ('T', ['T', 'T', 'T']),
        ('H', ['H', 'H', 'H']),
        ('T', ['T', 'T', 'T']),
        ('H', ['H', 'H', 'H']),
    ]
    
    unlabeled_data = []
    for labeled_doc in test_data:
        doc_contents = labeled_doc[1]
        unlabeled_data.append(doc_contents)
    
    em.init(2, unlabeled_data)
    
    
    #~ clustered_data = em.cluster(unlabeled_data, verbose=True)
    #~ 
    #~ print test_data
    #~ print clustered_data
    #~ 
    #~ results = {}
    #~ for r in zip(test_data, clustered_data):
        #~ r = (r[0][0], r[1])
        #~ results[r] = results.setdefault(r, 0) + 1
    #~ 
    #~ print results






DEBUG = False
REGEX = r"([^\W0-9]|['_])+"
COMPILED_REGEX = re.compile(REGEX, re.UNICODE)

# strip away metadata
def get_content(text):
    t = text.split('\n\n', 1)
    return t[1]

# turn into token sequence
def tokenize(text, stopwords=set()):
    features = []
    for match in COMPILED_REGEX.finditer(text):
        token = match.group().lower()
        if token not in stopwords:
            features.append(token)
    return features

SEED = 0
random.seed(SEED)
def reservoir_sampling(size, sample):
    sample_indices = []
    
    for index in xrange(size):
        # Generate the reservoir
        if index < sample:
            sample_indices.append(index)
        else:
            # Randomly replace elements in the reservoir
            # with a decreasing probability.             
            # Choose an integer between 0 and index (inclusive)               
            r = random.randint(0, index)               
            if r < sample:                       
                sample_indices[r] = index
    return set(sample_indices)

def newsgroups_dataset_iterator(path, stopwords, sample=50):
    for root, dirs, files in os.walk(path):
        if root == path:
            continue
        d = root.split('/', 1)[1]
        group = (d.split('.', 1)[0], d)
        indices = reservoir_sampling(len(files), sample)
        for index, file_name in enumerate(files):
            if index not in indices:
                continue
            with codecs.open(os.path.join(root, file_name), 'r', 'utf-8', errors='ignore') as f:
                text = f.read()
                content = get_content(text)
                seq = tokenize(content, stopwords)
                yield group, seq
    raise StopIteration

def prune(documents, top_n=10):
    doc_count = len(documents)
    dt_freq = [] # document word type frequency
    d_freq = {} # document frequency of type
    # populate counts
    for doc in documents:
        temp = {}
        for token in doc:
            temp[token] = temp.setdefault(token, 0) + 1
        for token in temp:
            d_freq[token] = d_freq.setdefault(token, 0) + 1
        dt_freq.append(temp)
    # calculate weights
    for t_freq in dt_freq:
        for token, count in t_freq.iteritems():
            weight = (1 + math.log(count))*math.log(doc_count/d_freq[token])
            t_freq[token] = weight
    if DEBUG: print(dt_freq[23])
    # find top n types per document
    allowed_words = set()
    for index, t_freq in enumerate(dt_freq):
        dt_freq[index] = dict(sorted(t_freq.iteritems(), key=lambda x: x[1], reverse=True)[:top_n])
        # compile list of permissible words
        for token in dt_freq[index]:
            allowed_words.add(token)
    result = []
    if DEBUG: print(dt_freq[23])
    for index, doc in enumerate(documents):
        temp = []
        for token in doc:
            if token in allowed_words:
                temp.append(token)
        result.append(temp)
    return result
    

def main():
    # get stopwords
    STOPWORD_FILE = 'english_all.txt'
    stopwords = set()
    with codecs.open(STOPWORD_FILE, 'r', 'utf-8') as f:
        for line in f:
            stopwords.add(line.strip().lower())
    
    print("Stopword Count:", str(len(stopwords)))
    
    # get documents as feature vectors
    DOCUMENTS_PATH = 'groups'
    dataset = newsgroups_dataset_iterator(DOCUMENTS_PATH, stopwords)
    documents = []
    document_classes = []
    major_classes = set()
    minor_classes = set()
    for doc_class, doc in dataset:
        document_classes.append(doc_class)
        documents.append(doc)
        major_classes.add(doc_class[0])
        minor_classes.add(doc_class[1])
    
    print("Doc Count:", str(len(documents)))
    print("Major Classes:", str(len(major_classes)))
    print("Minor Classes:", str(len(minor_classes)))
    
    if DEBUG: print(documents[23])
    # prune features
    pruned_documents = prune(documents)
    if DEBUG: print(pruned_documents[23])
    
    em = EM()
    # initialize
    em.init(7, pruned_documents)
    # cluster
    clustered_docs = em.cluster()
    

if __name__ == "__main__":
    main2()
