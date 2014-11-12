from __future__ import division, print_function

import os
import re
import sys
import math
import numpy as np
import codecs
import random


class EM(object):
    def __init__(self, likelihood_threshold=10**-2):
        self.threshold = likelihood_threshold
        self.previous_likelihood = float("-inf")
    
    def _log_likelihood_of_data(self):
        result = 
        return result
    
    def _check_threshold(self):
        
    
    def init(self, clusters, data):
        # transform to counts and get vocabulary
        self.clusters = clusters
        processed_data = []
        vocab = set()
        for doc in data:
            temp = {}
            for tok in doc:
                temp[tok] = temp.setdefault(tok, 0) + 1
                vocab.add(tok)
            processed_data.append(temp)
        self.processed_data = processed_data
        self.vocab = vocab
        
        # get initial lambdas
        self.lambdas = [1/clusters for i in xrange(clusters)]
        # get initial betas
        self.betas = np.random.dirichlet([0.1 for i in xrange(len(vocab))], clusters)
        
        
        # conduct EM
        while self._check_threshold():
            
        
    def cluster(self, data, table=stdout, verbose=False):
        # conduct EM
        while self.guard(self):
            p_class_given_doc = [] # [ [prob, prob]
            for doc in processed_data:
                total = 0
                temp = []
                for cluster, l in enumerate(lamb):
                    b = betas[cluster]
                    top = l
                    for t in doc:
                        top = top*(b[t]**doc[t])
                    temp.append(top)
                    total += top
                for index, entry in enumerate(temp):
                    temp[index] = entry/total
                p_class_given_doc.append(temp)
            
            # Print line in table
            if verbose:
                # Print line for Question 1
                row = []
                row.extend([self.iter_count, lamb[0], betas[0]['H'], betas[1]['H']])
                for entry in p_class_given_doc:
                    row.append(entry[0])
                table.write(','.join([str(e) for e in row]))
                table.write('\n')
            
            partial_counts = {}
            total_count = 0
            for cluster in xrange(len(lamb)):
                cluster_total = 0
                for t in types:
                    key = (t, cluster)
                    partial_count = 0
                    for index, p in enumerate(p_class_given_doc):
                        partial_count += p[cluster]*processed_data[index].setdefault(t, 0)
                    partial_counts[key] = partial_count
                    total_count += partial_count
                    cluster_total += partial_count
                    betas[cluster][t] = partial_count
                for b_key, b_value in betas[cluster].iteritems():
                    betas[cluster][b_key] = b_value/cluster_total
                lamb[cluster] = cluster_total
            
            for index, l in enumerate(lamb):
                lamb[index] = l/total_count
            
            self.iter_count += 1
        
        # make assignments to clusters
        result = []
        for doc in processed_data:
            max_cluster = 0
            max_value = -sys.float_info.max
            for cluster in xrange(self.clusters):
                b = betas[cluster]
                value = lamb[cluster]
                for t, count in doc.iteritems():
                    value *= (b[t]**count)
                if value > max_value:
                    max_value = value
                    max_cluster = cluster
            result.append(max_cluster)
        return result





















def main():
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
    #~ 
    #~ 
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






#~ DEBUG = False
#~ REGEX = r"([^\W0-9]|['_])+"
#~ COMPILED_REGEX = re.compile(REGEX, re.UNICODE)
#~ 
#~ # strip away metadata
#~ def get_content(text):
    #~ t = text.split('\n\n', 1)
    #~ return t[1]
#~ 
#~ # turn into token sequence
#~ def tokenize(text, stopwords=set()):
    #~ features = []
    #~ for match in COMPILED_REGEX.finditer(text):
        #~ token = match.group().lower()
        #~ if token not in stopwords:
            #~ features.append(token)
    #~ return features
#~ 
#~ SEED = 0
#~ random.seed(SEED)
#~ def reservoir_sampling(size, sample):
    #~ sample_indices = []
    #~ 
    #~ for index in xrange(size):
        #~ # Generate the reservoir
        #~ if index < sample:
            #~ sample_indices.append(index)
        #~ else:
            #~ # Randomly replace elements in the reservoir
            #~ # with a decreasing probability.             
            #~ # Choose an integer between 0 and index (inclusive)               
            #~ r = random.randint(0, index)               
            #~ if r < sample:                       
                #~ sample_indices[r] = index
    #~ return set(sample_indices)
#~ 
#~ def newsgroups_dataset_iterator(path, stopwords, sample=50):
    #~ for root, dirs, files in os.walk(path):
        #~ if root == path:
            #~ continue
        #~ d = root.split('/', 1)[1]
        #~ group = (d.split('.', 1)[0], d)
        #~ indices = reservoir_sampling(len(files), sample)
        #~ for index, file_name in enumerate(files):
            #~ if index not in indices:
                #~ continue
            #~ with codecs.open(os.path.join(root, file_name), 'r', 'utf-8', errors='ignore') as f:
                #~ text = f.read()
                #~ content = get_content(text)
                #~ seq = tokenize(content, stopwords)
                #~ yield group, seq
    #~ raise StopIteration
#~ 
#~ def prune(documents, top_n=6):
    #~ doc_count = len(documents)
    #~ dt_freq = [] # document word type frequency
    #~ d_freq = {} # document frequency of type
    #~ # populate counts
    #~ for doc in documents:
        #~ temp = {}
        #~ for token in doc:
            #~ temp[token] = temp.setdefault(token, 0) + 1
        #~ for token in temp:
            #~ d_freq[token] = d_freq.setdefault(token, 0) + 1
        #~ dt_freq.append(temp)
    #~ # calculate weights
    #~ for t_freq in dt_freq:
        #~ for token, count in t_freq.iteritems():
            #~ weight = (1 + math.log(count))*math.log(doc_count/d_freq[token])
            #~ t_freq[token] = weight
    #~ if DEBUG: print(dt_freq[23])
    #~ # find top n types per document
    #~ allowed_words = set()
    #~ for index, t_freq in enumerate(dt_freq):
        #~ dt_freq[index] = dict(sorted(t_freq.iteritems(), key=lambda x: x[1], reverse=True)[:top_n])
        #~ # compile list of permissible words
        #~ for token in dt_freq[index]:
            #~ allowed_words.add(token)
    #~ result = []
    #~ if DEBUG: print(dt_freq[23])
    #~ for index, doc in enumerate(documents):
        #~ temp = []
        #~ for token in doc:
            #~ if token in allowed_words:
                #~ temp.append(token)
        #~ result.append(temp)
    #~ return result
    #~ 
#~ 
#~ def main():
    #~ # get stopwords
    #~ STOPWORD_FILE = 'english_all.txt'
    #~ stopwords = set()
    #~ with codecs.open(STOPWORD_FILE, 'r', 'utf-8') as f:
        #~ for line in f:
            #~ stopwords.add(line.strip().lower())
    #~ 
    #~ print("Stopword Count:", str(len(stopwords)))
    #~ 
    #~ # get documents as feature vectors
    #~ DOCUMENTS_PATH = 'groups'
    #~ dataset = newsgroups_dataset_iterator(DOCUMENTS_PATH, stopwords)
    #~ documents = []
    #~ document_classes = []
    #~ major_classes = set()
    #~ minor_classes = set()
    #~ for doc_class, doc in dataset:
        #~ document_classes.append(doc_class)
        #~ documents.append(doc)
        #~ major_classes.add(doc_class[0])
        #~ minor_classes.add(doc_class[1])
    #~ 
    #~ print("Doc Count:", str(len(documents)))
    #~ print("Major Classes:", str(len(major_classes)))
    #~ print("Minor Classes:", str(len(minor_classes)))
    #~ 
    #~ if DEBUG: print(documents[23])
    #~ # prune features
    #~ pruned_documents = prune(documents)
    #~ if DEBUG: print(pruned_documents[23])
    #~ 
    #~ em = EM()
    #~ # initialize
    #~ em.init(documents)
    #~ clusters = 7 # there are 7 major categories found in the data
    #~ # clusters = 20 # there are 20 newsgroups as found in the data
    #~ lambdas = [1/clusters for i in xrange(clusters)] # initialize uniformly
    #~ alphas = [0.1 for i in len_vocab] # hyper parameters for beta
    #~ 
    #~ # cluster
    

if __name__ == "__main__":
    main()
