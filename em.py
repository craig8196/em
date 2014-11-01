from __future__ import division
import sys
from sys import stdout



class EM(object):
    def __init__(self):
        self.clusters = 2
        def guard(em): # indicates when we're done iterating
            if em.iter_count == 4:
                return False
            else:
                return True
        self.guard = guard
        self.lamb = [0.3, 0.7]
        self.beta = [
            { 'H': 0.3, 'T': 0.7 },
            { 'H': 0.6, 'T': 0.4 },
        ]
    
    def set_clusters(self, k):
        self.clusters = k
    
    # the guard determines whether or not we keep iterating
    def set_guard(self, function):
        self.guard = function
    
    def set_lambdas(self, lambdas):
        self.lamb = lambdas
    
    def set_betas(self, betas):
        self.beta = betas
    
    def cluster(self, data, table=stdout, verbose=False):
        # preprocess the data
        types = {}
        processed_data = [] # stores counts of each type
        for doc in data:
            temp = {}
            for w in doc:
                types[w] = True # types.setdefault(w, 0) + 1
                temp[w] = temp.setdefault(w, 0) + 1
            processed_data.append(temp)
        
        
        # initialize variables
        self.iter_count = 0
        
        # init lambdas
        lamb = []
        for l in self.lamb:
            lamb.append(l)
        
        # init betas
        betas = []
        for beta in self.beta:
            temp = {}
            for b, value in beta.iteritems():
                temp[b] = value
            betas.append(temp)
        
        #~ print "Initial Lambda:\n", lamb
        #~ print "Initial Beta:\n", betas
        #~ print "Raw Data:\n", data
        #~ print "Processed Data:\n", processed_data
        #~ print "Types:\n", types
        
        assert len(lamb) == self.clusters
        assert len(beta) == self.clusters
        for b in self.beta:
            assert len(b) == len(types)
        
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
            #~ print "Prob of a class given a doc:\n", p_class_given_doc
            
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
            #~ print "Total Count:\n", total_count
            #~ print "Partial Counts:\n", partial_counts
            
            for index, l in enumerate(lamb):
                lamb[index] = l/total_count
            #~ print "Lambdas:\n", lamb
            #~ print "Betas:\n", betas
            
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
                #~ print value, cluster
                if value > max_value:
                    max_value = value
                    max_cluster = cluster
            #~ print max_value, max_cluster
            result.append(max_cluster)
        return result





















def main():
    print 'Welcome'
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
    
    clustered_data = em.cluster(unlabeled_data, verbose=True)
    
    print test_data
    print clustered_data
    
    results = {}
    for r in zip(test_data, clustered_data):
        r = (r[0][0], r[1])
        results[r] = results.setdefault(r, 0) + 1
    
    print results


if __name__ == "__main__":
    main()
