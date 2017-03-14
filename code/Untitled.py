
# coding: utf-8

# In[2]:

import numpy as np
from big_clam import BigClam
from scipy.special import gamma, digamma, gammaln
import networkx as nx
import os, cPickle
from matplotlib import pyplot as plt
from Extends import draw_groups, NMI, GetComms, GetCommsPrev
from Experiments import draw_matrix
from Experiments import *
from big_clam_gamma import BigClamGamma

get_ipython().magic(u'matplotlib inline')


# In[24]:

def generate(power = 0.05):
    F_true = Fs3[0]
    A = gamma_model_test_data(F_true)
    P = 1 - np.exp(- power * A)
    rand = np.random.rand(*A.shape)
    mask = P <= (rand + rand.T) / 2

    B = A.copy()
    B[mask] = 0
    C = B.copy()
    C[B != 0] = 1
    return F_true, A, B, C

F_true, A, B, C = generate()
draw_test_sample(F_true, A, B, C)


# In[47]:

w_model3r = BigClamGamma(B, 3, debug_output=False, LLH_output=False, initF='cond_new_randz_spr', iter_output=1, processesNo=1,
                         rand_init_coef=0.5, stepSizeMod="backtracking", pow=g)
F_model3r, LLH3r = w_model3r.fit()
plt.figure(figsize=(15,15))
y = np.array(w_model3r.LLH_output_vals)
y = -y + y[-1]
plt.semilogy(y)


# In[45]:

w_model3r = BigClamGamma(B, 3, debug_output=False, LLH_output=False, initF='cond_new_randz_spr', iter_output=1, processesNo=1,
                         rand_init_coef=0.5, stepSizeMod="simple", pow=g)
F_model3r, LLH3r = w_model3r.fit()
plt.figure(figsize=(15,15))
y = np.array(w_model3r.LLH_output_vals)
y = -y + y[-1]
plt.semilogy(y)


# In[37]:

dir(w_model3r)


# In[3]:

w_model3r = BigClamGamma(B, 3, debug_output=False, LLH_output=True, initF='cond_new_randz_spr', iter_output=1000, processesNo=1,
                         rand_init_coef=0.5, stepSizeMod="simple", pow=0.05)
F_model3r, LLH3r = w_model3r.fit()

draw_res(B, F_true, F_model3r)
plt.show()


# In[17]:

pows = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
gammas = [0.001, 0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.5, 0.8, 1, 2]
Comm_True = GetComms(F_true.T, B)


# In[18]:

res = []
for g in log_progress(gammas):
    res.append([])
    for p in pows:
        F_true, A, B, C = generate(p)
        w_model3r = BigClamGamma(B, 3, debug_output=False, LLH_output=False, initF='cond_new_randz_spr', iter_output=1000, processesNo=1,
                                 rand_init_coef=0.5, stepSizeMod="simple", pow=g)
        F_model3r, LLH3r = w_model3r.fit()
        comms = GetComms(F_model3r, B)
        nmi = NMI(comms, None, Comm_True)
        res[-1].append(nmi)


# In[1]:

import seaborn as sns
#sns.set_style('white')


# In[21]:

plt.figure(figsize=(10,7))
sns.heatmap(res, xticklabels=pows, yticklabels=gammas)


# In[22]:

res


# In[ ]:



