
# coding: utf-8

# In[ ]:

import os
import cPickle
import scipy
import numpy as np

from TransE_functions import Embeddings, LayerTrans, Unstructured, parse_embeddings


# In[ ]:

f = open('../model/best_valid_model.pkl', 'rb')
loaded_objects = []
for i in range(3):
    loaded_objects.append(cPickle.load(f))
f.close()

loaded_objects[0]


# In[ ]:

embedding, relationl, relationr = parse_embeddings(loaded_objects[0])


# In[ ]:

ent2idx = open('../data/data_pkls/FB15k_entity2idx.pkl', 'rb')
entities = cPickle.load(ent2idx)


# ## - Now the calculations of distances between protein pairs:

# In[ ]:

#molecules_pairs_file = open("TransE_results_1-2.txt", 'r')
#molecules_pairs_file = open("Dump1_Dump2_ALLvsALL_20000_random_samples.txt", 'r')
molecules_pairs_file = open("TransE_500x500_1-2.txt", 'r')
molecules_pairs = molecules_pairs_file.readlines()
molecules_pairs_file.close()

print len(molecules_pairs)

results_distances = open('500_vs_500_TransE_results_20161110_1-2.txt', 'w')

count = 0

for line in molecules_pairs:
    count+=1
    if(count%50 == 0):
       print count 
    line = line.strip()
    left_p, right_p = line.split('\t')
    left_p_idx = entities[left_p]
    right_p_idx = entities[right_p]
    current_dis = 1/(1+scipy.spatial.distance.euclidean(embedding.E.get_value()[:,left_p_idx], embedding.E.get_value()[:,right_p_idx]))
    results_distances.write(left_p+'\t'+right_p+'\t'+str(current_dis)+"\n")

results_distances.close()

