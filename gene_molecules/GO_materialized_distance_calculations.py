
# coding: utf-8

# In[ ]:

import os
import cPickle
import scipy
import numpy as np

from TransE_functions import Embeddings, LayerTrans, Unstructured, parse_embeddings


# In[ ]:

dir_prefix = '../older/8th_2008_partiallyM_marge_0_1'

f = open( dir_prefix + '/model/best_valid_model.pkl', 'rb')
loaded_objects = []
for i in range(3):
    loaded_objects.append(cPickle.load(f))
f.close()

loaded_objects[0]

# In[ ]:

embedding, relationl, relationr = parse_embeddings(loaded_objects[0])

# In[ ]:

ent2idx = open( dir_prefix + '/data/data_pkls/FB15k_entity2idx.pkl', 'rb')
entities = cPickle.load(ent2idx)


# ## - Now the calculations of distances between protein pairs:

# In[ ]:

molecules_pairs_file = open( dir_prefix + '/data/original_data/proteinpairs2008.txt', 'r')
molecules_pairs = molecules_pairs_file.readlines()
molecules_pairs_file.close()

print len(molecules_pairs)

#results_distances = open('GO_materialized_2008_results.txt', 'w')
results_distances = open('GO_partially_materialized_marge_0_1_2008_results.txt', 'w')

count = 0

for line in molecules_pairs:
    count+=1
    if(count%500 == 0):
       print count 
    line = line.strip()
    left_p, right_p = line.split('\t')
    left_p_idx = entities['http://purl.org/obo/owl/GO#'+left_p]
    right_p_idx = entities['http://purl.org/obo/owl/GO#'+right_p]
    #NOW FOR 2014
    #left_p_idx = entities['http://purl.obolibrary.org/obo/'+left_p]
    #right_p_idx = entities['http://purl.obolibrary.org/obo/'+right_p]
    current_dis = 1/(1+scipy.spatial.distance.euclidean(embedding.E.get_value()[:,left_p_idx], embedding.E.get_value()[:,right_p_idx]))
    results_distances.write(left_p+'\t'+right_p+'\t'+str(current_dis)+"\n")

results_distances.close()

