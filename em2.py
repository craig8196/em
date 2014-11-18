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
        self.max_iter = 10
        self.curr_iter = 0
    
    def _log_likelihood_of_data(self):
        return self.likelihoods.sum()
    
    def _check_threshold(self):
        #~ self.curr_iter += 1
        #~ if self.curr_iter > self.max_iter:
            #~ return False
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
            
            #~ for i in xrange(clusters): # make sure betas don't violate laws of probability
                #~ print(np.exp(new_betas[i]).sum())
            
            self.lambdas = new_lambdas
            self.betas = new_betas
        
    def cluster(self, data=None):
        is_processed = False
        if data is None:
            is_processed = True
            data = self.processed_data
        
        if not is_processed:
            pass # TODO process data
        
        results = []
        for doc in data:
            max_cluster = 0
            max_prob = float('-inf')
            for k in xrange(self.clusters):
                logprob = self.betas[k].dot(doc)
                logprob += self.lambdas[k]
                if logprob > max_prob:
                    max_prob = logprob
                    max_cluster = k
            results.append(max_cluster)
        return results

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
    print(em.cluster())


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

def newsgroups_dataset_iterator(path, stopwords, sample=100):
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
    # remove singletons
    singletons = set()
    for tok, cnt in d_freq.iteritems():
        if cnt == 1:
            singletons.add(tok)
    # calculate weights
    for t_freq in dt_freq:
        for token, count in t_freq.iteritems():
            if token in singletons:
                weight = 0
            else:
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

def confusion_matrix_to_file(conf_matrix, file_name):
    """Takes a dict conf_matrix where keys are 2-tuples and they map to counts."""
    rows = set()
    cols = set()
    for key in conf_matrix:
        rows.add(key[0])
        cols.add(key[1])
    rows = sorted(list(rows))
    cols = sorted(list(cols))
    
    matrix = [[0 for i in xrange(len(cols))] for i in xrange(len(rows))]
    
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            matrix[i][j] = conf_matrix.setdefault((row, col), 0)
    
    result = ',' + ','.join(map(lambda x: str(x), cols)) + '\n'
    for i, row in enumerate(rows):
        result += str(row) + ','
        result += ','.join(map(lambda x: str(x), matrix[i]))
        result += '\n'
    
    with codecs.open(file_name, 'w', 'utf-8') as f:
        f.write(result)
    return result

def compute_ari(d):
    """d -- Dict with a tuple as the key and count as the value"""
    def choose_2(n_top):
        if n_top < 2:
            return 0
        return (n_top)*(n_top-1)/2
    
    rows = {}
    cols = {}
    n = 0 # total count
    for t, count in d.iteritems():
        row = t[0]
        col = t[1]
        rows[row] = rows.setdefault(row, 0) + count
        cols[col] = cols.setdefault(col, 0) + count
        n += count
    print(rows)
    print(cols)
    rows = dict(map(lambda(k,v):(k, choose_2(v)), rows.iteritems()))
    cols = dict(map(lambda(k,v):(k, choose_2(v)), cols.iteritems()))
    ROWS = sum(rows.values()) # sum over rows
    COLS = sum(cols.values()) # sum over columns
    A = sum(map(lambda(k,v): choose_2(v), d.iteritems()))
    B = ROWS*COLS
    C = choose_2(n)
    D = ROWS+COLS
    return (A-B/C)/(D/2 - B/C)

def test_ari():
    d = {
        (1, 1): 2,
        (3, 2): 2,
        (2, 3): 2,
        (1, 2): 0,
    }
    print(compute_ari(d)) # should be 1.0
    d = {
        (1, 1): 1,
        (1, 2): 1,
        (2, 3): 2,
        (3, 1): 1,
        (3, 2): 1,
    }
    print(compute_ari(d)) # should be ~ 1/6

def test_confusion_to_file():
    d = {
        (1, 1): 2,
        (3, 2): 2,
        (2, 3): 2,
        (1, 2): 0,
    }
    print(confusion_matrix_to_file(d, 'test1.csv'))
    d = {
        (1, 1): 1,
        (1, 2): 1,
        (2, 3): 2,
        (3, 1): 1,
        (3, 2): 1,
    }
    print(confusion_matrix_to_file(d, 'test2.csv'))

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
    #~ em.init(7, pruned_documents)
    em.init(20, pruned_documents)
    # cluster
    clustered_docs = em.cluster()
    
    classes = set()
    results = {}
    for mine, actual in zip(clustered_docs, document_classes):
        t = (mine, actual[1])
        classes.add(actual[1])
        results[t] = results.setdefault(t, 0) + 1
    print(results)
    for c in classes:
        print('Class:', c)
        for k, v in results.iteritems():
            if k[1] == c:
                print(k[0], v)

if __name__ == "__main__":
    test_confusion_to_file()
