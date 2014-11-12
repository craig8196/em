from __future__ import division, print_function

import os
import re
import sys
import math
import numpy
import scipy
import codecs
import random

REGEX = r"([^\W]|['_])+"
COMPILED_REGEX = re.compile(REGEX, re.UNICODE)

class Pruner(object):
    
    # keep -- number of words to keep per document
    def __init__(self, keep=10):
        self.keep = keep
        pool = None
    
    def prune(self, documents):
        pass
    

class DatasetIterator(object):
    
    def __init__(self, path):
        doc_paths = []
        for asdf in os.walk(path):
            pass



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

def main():
    # get stopwords
    STOPWORD_FILE = 'english_all.txt'
    stopwords = set()
    with codecs.open(STOPWORD_FILE, 'r', 'utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    
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
    
    
    # prune features
    pass
    
    
    
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
