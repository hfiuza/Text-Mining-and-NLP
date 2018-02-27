import string
import re 
import itertools
import copy
import igraph
import nltk

from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag

def clean_text_simple(text, remove_stopwords=True, pos_filtering=True, stemming=True):
    
    punct = string.punctuation.replace('-', '')
    
    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    text = ''.join(l for l in text if l not in punct)
    # strip extra white space
    text = re.sub(' +',' ',text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    ### fill the gap ###
    tokens = text.split(' ')
    if pos_filtering == True:
        # apply POS-tagging
        tagged_tokens = pos_tag(tokens)
        # retain only nouns and adjectives
        tokens_keep = []
        for i in range(len(tagged_tokens)):
            item = tagged_tokens[i]
            if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR'
            ):
                tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        stpwds = stopwords.words('english')
        # remove stopwords
        ### fill the gap ###
        tokens = [ token for token in tokens if token not in set(stpwds) ]

    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed

    return(tokens)

	
def terms_to_graph(terms, w):
    # This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    # Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'
    
    from_to = {}
    
    # create initial graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))
    
    new_edges = []
    
    for i in range(len(indexes)):
        new_edges.append(' '.join(list(terms_temp[i] for i in indexes[i])))
    # weird list comprehension       
    for i in range(0,len(new_edges)):
        from_to[new_edges[i].split()[0],new_edges[i].split()[1]] = 1

    # then iterate over the remaining terms
    for i in xrange(w, len(terms)):
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i-w+1):(i+1)]
        
        # edges to try
        candidate_edges = []
        for p in xrange(w-1):
            candidate_edges.append((terms_temp[p],considered_term))
    
        for try_edge in candidate_edges:
            
            # if not self-edge
            if try_edge[1] != try_edge[0]:
            
                # if edge has already been seen, update its weight
                ### fill the gap ###
                if try_edge in from_to.keys():
                    from_to[try_edge] += 1
                                   
                # if edge has never been seen, create it and assign it a unit weight     
                else:
                    ### fill the gap ###
                    from_to[try_edge] = 1
    
    # create empty graph
    g = igraph.Graph(directed=True)
    
    # add vertices
    g.add_vertices(sorted(set(terms)))
    
    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())
    
    # set edge and vertice weights
    g.es['weight'] = from_to.values() # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=from_to.values()) # weighted degree
    
    return(g)


def unweighted_k_core(g):
    # work on clone of g to preserve g 
    gg = copy.deepcopy(g)
    
    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs['name'],[0]*len(gg.vs)))
    i = 0

    # while there are vertices remaining in the graph
    while len(gg.vs)>0:
        ### fill the gaps ###
        # print 4
        # print i, len(gg.vs)
        degrees = gg.strength()
        while len(degrees) > 0 and min(degrees) <= i:
            to_delete_ids = []
            for ind, degree in enumerate(degrees):
                if degree <= i:
                    to_delete_ids.append(ind)
                    cores_g[gg.vs[ind]['name']] = i
            # print 'to_delete_ids:'
            # print to_delete_ids
            gg.delete_vertices(to_delete_ids)
            degrees = gg.strength()
        # update core number for remaining vertices
        i += 1
    print cores_g.values()

    return cores_g
	

def accuracy_metrics(candidate, truth):
    
    # true positives ('hits') are both in candidate and in truth
    tp = len(set(candidate).intersection(truth))
    
    # false positives ('false alarms') are in candidate but not in truth
    fp = len([element for element in candidate if element not in truth])
    
    # false negatives ('misses') are in truth but not in candidate
    ### fill the gap ###
    fn = len([element for element in truth if element not in candidate])

    # precision
    prec = round(float(tp)/(tp+fp),5)
    
    # recall
    ### fill the gap ###
    rec = round(float(tp)/(tp+fn), 5) 
   
    if prec+rec != 0:
        # F1 score
        f1 = round(2 * float(prec*rec)/(prec+rec),5)
    else:
        f1=0
       
    return (prec, rec, f1)