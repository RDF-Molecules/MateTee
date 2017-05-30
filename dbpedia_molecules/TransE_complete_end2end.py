
# coding: utf-8

# ## 1- Transform the data:

# In[1]:

import os
import cPickle
import glob
import sys
import numpy as np
import scipy.sparse as sp
import time
import copy

import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
from collections import OrderedDict
import rdflib

dump_base_path = '.'

transformed_f = open('data/data_transformed/nt_dump_2_triples.txt', 'w')

for dump_number in range(3):
    print 'Dump '+str(dump_number)+':'
    g=rdflib.Graph()
    g.load('data/original_data/fuhsen_molecules/materialized-knowledge-dump%d.nt' % dump_number, format="nt")

    print len(g)

    count = 1

    for s,p,o in g:    
       if 'www.ontologydesignpatterns.org' not in p:
           transformed_f.write(s.encode('utf-8')+'/dump'+str(dump_number)+'\t'+p.encode('utf-8')+'\t'+o.replace('\t', 'TAB_WAS_HERE').replace('\n', 'CR_WAS_HERE').encode('utf-8')+'\n')
           if count%10000 == 0:
               print count
           count+=1

transformed_f.close()

# In[2]:

f_data_split = open(dump_base_path+'/data/data_transformed/nt_dump_2_triples.txt', 'r')
data_split = f_data_split.readlines()
f_data_split.close()

from random import shuffle

shuffle(data_split)

total_triples = len(data_split)

#train_new = data_split[:int(total_triples*0.6)]
train_new = data_split[:]
test_new = data_split[int(total_triples*0.6):int(total_triples*0.8)]
val_new = data_split[int(total_triples*0.8):]

print total_triples
train_new


# In[3]:

print len(train_new)
print len(test_new)
print len(val_new)


# In[4]:

with open(dump_base_path+'/data/data_transformed/freebase_mtr100_mte100-train.txt', 'w') as text_file:
    for x in train_new:
        text_file.write(x)


# In[5]:

with open(dump_base_path+'/data/data_transformed/freebase_mtr100_mte100-test.txt', 'w') as text_file:
    for x in test_new:
        text_file.write(x)


# In[6]:

with open(dump_base_path+'/data/data_transformed/freebase_mtr100_mte100-valid.txt', 'w') as text_file:
    for x in val_new:
        text_file.write(x)


# ## 2- Preprocess the data:

# In[7]:

# Put the freebase15k data absolute path here
#datapath = '/home/camilo/IPython_notebooks/TransE_embeddings/Data_FB15k/FB15k/'
datapath = dump_base_path+'/data/data_transformed/'

def parseline(line):
    lhs, rel, rhs = line.split('\t')
    lhs = lhs.split(' ')
    rhs = rhs.split(' ')
    rel = rel.split(' ')
    return lhs, rel, rhs


# In[8]:

#################################################
### Creation of the entities/indices dictionnaries

np.random.seed(753)

entleftlist = []
entrightlist = []
rellist = []

for datatyp in ['train']:
    f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()
    #count = 0
    for i in dat:
        #print "Line: " + str(count) + ", value: " + i[:-1]
        lhs, rel, rhs = parseline(i[:-1])
        entleftlist += [lhs[0]]
        entrightlist += [rhs[0]]
        rellist += [rel[0]]
        #count+=1

entleftset = np.sort(list(set(entleftlist) - set(entrightlist)))
entsharedset = np.sort(list(set(entleftlist) & set(entrightlist)))
entrightset = np.sort(list(set(entrightlist) - set(entleftlist)))
relset = np.sort(list(set(rellist)))

np.set_printoptions(threshold=np.inf)
print relset

entity2idx = {}
idx2entity = {}

# we keep the entities specific to one side of the triplets contiguous
idx = 0
for i in entrightset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbright = idx
for i in entsharedset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbshared = idx - nbright
for i in entleftset:
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbleft = idx - (nbshared + nbright)

print "# of only_left/shared/only_right entities: ", nbleft, '/', nbshared, '/', nbright
# add relations at the end of the dictionary

for i in relset:
    #print i
    entity2idx[i] = idx
    idx2entity[idx] = i
    idx += 1
nbrel = idx - (nbright + nbshared + nbleft)
print "Number of relations: ", nbrel

f = open(dump_base_path+'/data/data_pkls/FB15k_entity2idx.pkl', 'w')
g = open(dump_base_path+'/data/data_pkls/FB15k_idx2entity.pkl', 'w')
cPickle.dump(entity2idx, f, -1)
cPickle.dump(idx2entity, g, -1)
f.close()
g.close()


# In[9]:

#################################################
### Creation of the dataset files

unseen_ents=[]
remove_tst_ex=[]

for datatyp in ['valid', 'train', 'test']:
    print datatyp
    f = open(datapath + 'freebase_mtr100_mte100-%s.txt' % datatyp, 'r')
    dat = f.readlines()
    f.close()

    # Declare the dataset variables
    inpl = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)), dtype='float32')
    inpr = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)), dtype='float32')
    inpo = sp.lil_matrix((np.max(entity2idx.values()) + 1, len(dat)), dtype='float32')
    # Fill the sparse matrices
    ct = 0
    for i in dat:
        lhs, rel, rhs = parseline(i[:-1])
        if lhs[0] in entity2idx and rhs[0] in entity2idx and rel[0] in entity2idx: 
            inpl[entity2idx[lhs[0]], ct] = 1
            inpr[entity2idx[rhs[0]], ct] = 1
            inpo[entity2idx[rel[0]], ct] = 1
            ct += 1
        else:
            if lhs[0] in entity2idx:
                unseen_ents+=[lhs[0]]
            if rel[0] in entity2idx:
                unseen_ents+=[rel[0]]
            if rhs[0] in entity2idx:
                unseen_ents+=[rhs[0]]
            remove_tst_ex+=[i[:-1]]

    # Save the datasets
    f = open(dump_base_path+'/data/data_pkls/FB15k-%s-lhs.pkl' % datatyp, 'w')
    g = open(dump_base_path+'/data/data_pkls/FB15k-%s-rhs.pkl' % datatyp, 'w')
    h = open(dump_base_path+'/data/data_pkls/FB15k-%s-rel.pkl' % datatyp, 'w')
    cPickle.dump(inpl.tocsr(), f, -1)
    cPickle.dump(inpr.tocsr(), g, -1)
    cPickle.dump(inpo.tocsr(), h, -1)
    f.close()
    g.close()
    h.close()

unseen_ents=list(set(unseen_ents))
print len(unseen_ents)
remove_tst_ex=list(set(remove_tst_ex))
print len(remove_tst_ex)

for i in remove_tst_ex:
    print i


# ## 3- Find the embeddings:

# In[10]:

datapath=dump_base_path+'/data/data_pkls/'
dataset='FB15k'
Nent=nbleft+nbshared+nbright+nbrel
rhoE=1
rhoL=5
Nsyn= nbleft+nbshared+nbright
Nrel=nbrel
loadmodel=False
loademb=False
op='TransE'
simfn='L2'
ndim=50
nhid=50
marge=0.5
lremb=0.01
lrparam=0.01
nbatches=100
totepochs=1000
#totepochs=100
test_all=1000
#test_all=10
neval=1000
seed=123
savepath=dump_base_path+'/model'
loadmodelBi=False
loadmodelTri=False


# In[11]:

class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z


# In[12]:

# Argument of the experiment script
state = DD()
state.datapath = datapath
state.dataset = dataset
state.Nent = Nent
state.Nsyn = Nsyn
state.Nrel = Nrel
state.loadmodel = loadmodel
state.loadmodelBi = loadmodelBi
state.loadmodelTri = loadmodelTri
state.loademb = loademb
state.op = op
state.simfn = simfn
state.ndim = ndim
state.nhid = nhid
state.marge = marge
state.rhoE = rhoE
state.rhoL = rhoL
state.lremb = lremb
state.lrparam = lrparam
state.nbatches = nbatches
state.totepochs = totepochs
state.test_all = test_all
state.neval = neval
state.seed = seed
state.savepath = savepath


# In[13]:

class Channel(object):
    def __init__(self, state):
        self.state = state
        f = open(self.state.savepath + '/orig_state.pkl', 'w')
        cPickle.dump(self.state, f, -1)
        f.close()
        self.COMPLETE = 1

    def save(self):
        f = open(self.state.savepath + '/current_state.pkl', 'w')
        cPickle.dump(self.state, f, -1)
        f.close()


# In[14]:

channel = Channel(state)

# Show experiment parameters
print state
np.random.seed(state.seed)


# In[15]:

def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)), dtype=theano.config.floatX) #Compressed Sparse Row matrix

def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]

class LayerTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of 
    of the 'left member' and 'right member'i.e. translating x by y.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x+y

class Unstructured(object):
    """
    Class for a layer with two input vectors that performs the linear operator
    of the 'left member'.

    :note: The 'right' member is the relation, therefore this class allows to
    define an unstructured layer (no effect of the relation) in the same
    framework.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x


# In[16]:

# Positives
trainl = load_file(state.datapath + state.dataset + '-train-lhs.pkl')
trainr = load_file(state.datapath + state.dataset + '-train-rhs.pkl')
traino = load_file(state.datapath + state.dataset + '-train-rel.pkl')

traino = traino[-state.Nrel:, :]


# In[17]:

# Valid set
validl = load_file(state.datapath + state.dataset + '-valid-lhs.pkl')
validr = load_file(state.datapath + state.dataset + '-valid-rhs.pkl')
valido = load_file(state.datapath + state.dataset + '-valid-rel.pkl')

valido = valido[-state.Nrel:, :]


# In[18]:

# Test set
testl = load_file(state.datapath + state.dataset + '-test-lhs.pkl')
testr = load_file(state.datapath + state.dataset + '-test-rhs.pkl')
testo = load_file(state.datapath + state.dataset + '-test-rel.pkl')

testo = testo[-state.Nrel:, :]


# In[19]:

# Index conversion
trainlidx = convert2idx(trainl)[:state.neval]
trainridx = convert2idx(trainr)[:state.neval]
trainoidx = convert2idx(traino)[:state.neval]

validlidx = convert2idx(validl)[:state.neval]
validridx = convert2idx(validr)[:state.neval]
validoidx = convert2idx(valido)[:state.neval]

testlidx = convert2idx(testl)[:state.neval]
testridx = convert2idx(testr)[:state.neval]
testoidx = convert2idx(testo)[:state.neval]

idxl = convert2idx(trainl)
idxr = convert2idx(trainr)
idxo = convert2idx(traino)

idxtl = convert2idx(testl)
idxtr = convert2idx(testr)
idxto = convert2idx(testo)

idxvl = convert2idx(validl)
idxvr = convert2idx(validr)
idxvo = convert2idx(valido)


# In[20]:

true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T


# In[21]:

# Embeddings class -----------------------------------------------------------
class Embeddings(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings.
        :param tag: name of the embeddings for parameter declaration.
        """
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D, N))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)
# ----------------------------------------------------------------------------


# In[22]:

# Model declarationpp state
leftop  = LayerTrans()
rightop = Unstructured()
    
# embeddings
embeddings = Embeddings(np.random, state.Nent, state.ndim, 'emb')


# In[23]:

#if state.op == 'TransE' and type(embeddings) is not list:
relationVec = Embeddings(np.random, state.Nrel, state.ndim, 'relvec')
embeddings = [embeddings, relationVec, relationVec]


# In[24]:

def L2sim(left, right):
    return - T.sqrt(T.sum(T.sqr(left - right), axis=1))

# Cost ------------------------------------------------------------------------
def margincost(pos, neg, marge=1.0):
    out = neg - pos + marge
    return T.sum(out * (out > 0)), out > 0


# In[25]:

simfn = eval(state.simfn + 'sim')


# In[26]:

def parse_embeddings(embeddings):
    """
    Utilitary function to parse the embeddings parameter in a normalized way
    for the Structured Embedding [Bordes et al., AAAI 2011] and the Semantic
    Matching Energy [Bordes et al., AISTATS 2012] models.
    """
    if type(embeddings) == list:
        embedding = embeddings[0]
        relationl = embeddings[1]
        relationr = embeddings[2]
    else:
        embedding = embeddings
        relationl = embeddings
        relationr = embeddings
    return embedding, relationl, relationr

def TrainFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0, rel=True):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    # Negative 'left' member
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    # Negative 'right' member
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    cost = costl + costr
    out = T.concatenate([outl, outr])
    # List of inputs of the function
    list_in = [lrembeddings, lrparams,
            inpl, inpr, inpo, inpln, inprn]
    if rel:
        # If rel is True, we also consider a negative relation member
        inpon = S.csr_matrix()
        relln = S.dot(relationl.E, inpon).T
        relrn = S.dot(relationr.E, inpon).T
        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))
        costo, outo = margincost(simi, simion, marge)
        cost += costo
        out = T.concatenate([out, outo])
        list_in += [inpon]

    if hasattr(fnsim, 'params'):
        # If the similarity function has some parameters, we update them too.
        gradientsparams = T.grad(cost,
            leftop.params + rightop.params + fnsim.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params + fnsim.params, gradientsparams))
    else:
        gradientsparams = T.grad(cost, leftop.params + rightop.params)
        updates = OrderedDict((i, i - lrparams * j) for i, j in zip(
            leftop.params + rightop.params, gradientsparams))
    gradients_embedding = T.grad(cost, embedding.E)
    newE = embedding.E - lrembeddings * gradients_embedding
    updates.update({embedding.E: newE})
    if type(embeddings) == list:
        # If there are different embeddings for the relation member.
        gradients_embedding = T.grad(cost, relationl.E)
        newE = relationl.E - lrparams * gradients_embedding
        updates.update({relationl.E: newE})
        gradients_embedding = T.grad(cost, relationr.E)
        newE = relationr.E - lrparams * gradients_embedding
        updates.update({relationr.E: newE})
    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
            updates=updates, on_unused_input='ignore')


# In[27]:

def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = rightop(rhs, relr)
    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi], on_unused_input='ignore')


# In[28]:

def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = leftop(lhs, rell)
    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')


# In[29]:

# Function compilation
trainfunc = TrainFn1Member(simfn, embeddings, leftop, rightop, marge=state.marge, rel=False)
ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)
rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop, subtensorspec=state.Nsyn)


# In[30]:

out = []
outb = []
state.bestvalid = -1

batchsize = trainl.shape[1] / state.nbatches


# In[31]:

def create_random_mat(shape, listidx=None):
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    randommat = scipy.sparse.lil_matrix((shape[0], shape[1]),
            dtype=theano.config.floatX)
    idx_term = 0
    for idx_ex in range(shape[1]):
        if idx_term == len(listidx):
            idx_term = 0
        randommat[listidx[idx_term], idx_ex] = 1
        idx_term += 1
    return randommat.tocsr()


# In[32]:

def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []

    count = 0
    
    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)
        
        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        scores_l = (sl(r, o)[0]).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]
        
        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
        
        count+=1
        if count % 500 == 0:
            print count
        
    return errl, errr


# In[33]:

print "BEGIN TRAINING"
timeref = time.time()
for epoch_count in xrange(1, state.totepochs + 1):
    print "Current iteration: " + str(epoch_count+1) + " , "
    # Shuffling
    order = np.random.permutation(trainl.shape[1])
    trainl = trainl[:, order]
    trainr = trainr[:, order]
    traino = traino[:, order]

    # Negatives
    trainln = create_random_mat(trainl.shape, np.arange(state.Nsyn))
    trainrn = create_random_mat(trainr.shape, np.arange(state.Nsyn))

    for i in range(state.nbatches):
        tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
        tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
        tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
        tmpnl = trainln[:, i * batchsize:(i + 1) * batchsize]
        tmpnr = trainrn[:, i * batchsize:(i + 1) * batchsize]
        # training iteration
        outtmp = trainfunc(state.lremb, state.lrparam, tmpl, tmpr, tmpo, tmpnl, tmpnr)
        out += [outtmp[0] / float(batchsize)]
        outb += [outtmp[1]]
        
        # embeddings normalization
        embeddings[0].normalize()

    if (epoch_count % state.test_all) == 0:
        # model evaluation
        print "-- EPOCH %s (%s seconds per epoch):" % (
                epoch_count,
                round(time.time() - timeref, 3) / float(state.test_all))
        timeref = time.time()
        print "COST >> %s +/- %s, %% updates: %s%%" % (
                round(np.mean(out), 4), round(np.std(out), 4),
                round(np.mean(outb) * 100, 3))
        out = []
        outb = []
        resvalid = FilteredRankingScoreIdx(ranklfunc, rankrfunc, validlidx, validridx, validoidx, true_triples)
        state.valid = np.mean(resvalid[0] + resvalid[1])
        restrain = FilteredRankingScoreIdx(ranklfunc, rankrfunc, trainlidx, trainridx, trainoidx, true_triples)
        state.train = np.mean(restrain[0] + restrain[1])
        print "\tMEAN RANK >> valid: %s, train: %s" % (state.valid, state.train)
        if state.bestvalid == -1 or state.valid < state.bestvalid:
            restest = FilteredRankingScoreIdx(ranklfunc, rankrfunc,testlidx, testridx, testoidx, true_triples)
            state.bestvalid = state.valid
            state.besttrain = state.train
            state.besttest = np.mean(restest[0] + restest[1])
            state.bestepoch = epoch_count
            # Save model best valid model
            f = open(state.savepath + '/best_valid_model.pkl', 'w')

            cPickle.dump(embeddings, f, -1)
            cPickle.dump(leftop, f, -1)
            cPickle.dump(rightop, f, -1)
            cPickle.dump(simfn, f, -1)
                
            f.close()
            print "\t\t##### NEW BEST VALID >> test: %s" % (state.besttest)
        
        # Save current model
        f = open(state.savepath + '/current_model.pkl', 'w')

        cPickle.dump(embeddings, f, -1)
        cPickle.dump(leftop, f, -1)
        cPickle.dump(rightop, f, -1)
        cPickle.dump(simfn, f, -1)
            
        f.close()
        state.nbepochs = epoch_count
        print "\t(the evaluation took %s seconds)" % (round(time.time() - timeref, 3))
        timeref = time.time()
        channel.save()
#return channel.COMPLETE
print "FINISHED"
channel.COMPLETE

