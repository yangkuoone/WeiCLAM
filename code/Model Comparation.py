
# coding: utf-8

# # Эксперименты на реальных данных, сравнение моделей

# In[1]:

from Extends import *

from big_clam import BigClam
from big_clam_gamma import BigClamGamma
from Experiments import *
from time import gmtime, strftime
from cPickle import dump, load
from multiprocessing import pool
import networkx as nx
from sklearn.decomposition import NMF

def time():
    return strftime("%H:%M:%S ", gmtime())

# In[2]:




# In[3]:

def NMF_clust(A, K):
    model = NMF(n_components=K)
    res = model.fit_transform(A)
    #print res.shape
    return res


# In[4]:


# # Генерация модельных примеров

# In[46]:

seed = 21113222
data_params = {
    'N': 1000,
     'mut': 0.1,
     'maxk': 50,
     'k': 30,
     'om': 2,
     'muw': 0.1,
     'beta': 2,
     't1': 2,
     't2': 2,
     #'minc': 100,
     'on': 0,
     }


model_params = {
    'initF': 'cond_new_randz',
    'LLH_output': False,
    'iter_output': 20000,
    'processesNo': 1,
    'dump': 1000,
    'eps': 1e-2,
    "max_iter": 500000,
}

# In[47]:

models = {#'BigClam-Zeros': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, **model_params).fit()[0],
          #'BigClam-Zeros-simple': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, stepSizeMod='simple', **model_params).fit()[0],
          #'BigClam-Mean': lambda A, K, name: BigClam(1.0 * (A < np.mean(A)), K, dump_name=name, **model_params).fit()[0],
          'BigClamWeighted': lambda A, K, name: BigClam(A, K, dump_name=name, **model_params).fit()[0],
          'SparseGamma': lambda A, K, name: BigClamGamma(A, K, dump_name=name, **model_params).fit()[0],
          'BigClam-orig-zeros': lambda A, K, name: bigclam_orig(1.0 * (A != 0), K),
          'BigClamWeighted-sp10': lambda A, K, name: BigClam(A, K, dump_name=name, sparsity_coef=10,  **model_params).fit()[0],
          'SparseGamma-sp10': lambda A, K, name: BigClamGamma(A, K, dump_name=name, sparsity_coef=10, **model_params).fit()[0],
          #'BigClam-orig-mean': lambda A, K, name: bigclam_orig(1.0 * (A < np.mean(A)), K),
          'COPRA': lambda A, K, name: copra(A,K),
          'NMF': lambda A, K, name: NMF_clust(A,K),
          #'CPM': lambda A, K, name: [list(x) for x in get_percolated_cliques(nx.from_numpy_matrix(1.0 * (A != 0)), 5)]
        }

qual_fun = {'MixedModularity': MixedModularity,
            '1-MeanConductance': lambda F,A: 1-MeanConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MeanConductance(F, A),
            '1-MaxConductance': lambda F,A: 1-MaxConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MaxConductance(F, A),
            'NMI': lambda F,A, true_comm: NMI(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm),
            #'NMI_new': lambda F,A, true_comm: NMI3(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm),
            }

# In[48]:

def worker(iter):
        res = {name: {key: 0 for key in qual_fun} for name in models}
        print ' {}:'.format(iter),
        G, comms = LancichinettiBenchmark(**data_params)
        A = np.array(nx.to_numpy_matrix(G))
        for name in models:
            print '!',
            F = models[name](A, len(comms), name)
            print '.',
            for key in qual_fun:
                try:
                    if key not in {"NMI", 'NMI_new'}:
                        q = qual_fun[key](F, A)
                    else:
                        q = qual_fun[key](F, A, comms)
                except:
                    print 'Some err in ' + name,
                    q = -1e-10
                res[name][key] = q
        return res


def mean(l):
    return 1.0 * sum(l) / len(l)


if __name__ == '__main__':
    Pool = pool.Pool(processes=5)

    iter_count = 10

    mixing_range = np.linspace(0, 0.7, 21)
    mixing_range = np.linspace(0, 0.7, 21)
    models_res = []
    (models_res, mixing_range, mix, data_params) = load(file('../data/dumps/models_res_temp-3-dump'))
    for i_mix, mix in enumerate(mixing_range):
        if i_mix < 4:
            continue
        print '{} mix: {}'.format(time(), mix)
        with file(r'..\external\Lancichinetti benchmark\time_seed.dat', 'w') as f:
            f.write(str(seed))
        data_params['on'] = np.floor(data_params['N'] * mix)
        one_graph_res = {name: {key: [] for key in qual_fun} for name in models}
        res = []
        for iter in xrange(iter_count):
            res.append(worker(iter))
        #res = Pool.map(worker, xrange(iter_count))
        for iter in xrange(iter_count):
            for name in one_graph_res:
                for key in one_graph_res[name]:
                    one_graph_res[name][key].append(res[iter][name][key])

        models_res.append(one_graph_res)
        dump((models_res, mixing_range, mix, data_params),
             file('../data/dumps/models_res_temp-{}-dump'.format(i_mix), 'w'))

    dump((models_res, mixing_range, mix, data_params), file('../data/dumps/models_res_full-dump', 'w'))

    # In[17]:

    (models_res, mixing_range, mix, data_params) = load(file('../data/dumps/models_res_full-dump'))


    # In[49]:


    plt.figure(figsize=(15, 10))
    for indx, qual_name in enumerate(qual_fun):
        plt.subplot(2, len(qual_fun) / 2, indx + 1)
        plt.ylabel('{}, N={}'.format(qual_name, data_params['N']))
        plt.xlabel('mixing parameter')
        colors = plt.get_cmap('hsv')(np.linspace(0, 1.0, len(models) + 1))
        for i, name in enumerate(models):
            plt.plot(mixing_range, [res[name][qual_name][0] for res in models_res if len(res) != 0], label=name,
                     color=colors[i])
        if indx == 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    print
    print
    for mix, res in zip(mixing_range, models_res):
        for key in res:
            print mix, ': ', key, res[key]
    plt.show()