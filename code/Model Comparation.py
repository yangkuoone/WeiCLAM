
# coding: utf-8

# # Эксперименты на реальных данных, сравнение моделей

# In[1]:

from Extends import *

from big_clam import BigClam
from big_clam_gamma import BigClamGamma
from Experiments import *
from time import gmtime, strftime
from cPickle import dump, load
from multiprocessing import Pool
import networkx as nx
from sklearn.decomposition import NMF
from shutil import copyfile

def time():
    return strftime("%H:%M:%S ", gmtime())

# In[2]:

save_all=False


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
    'N': 400,
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
    'dump': False,
    'eps': 1e-2,
    "max_iter": 500000,
}

# In[47]:

models = {#'BigClam-Zeros': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, **model_params).fit()[0],
          #'BigClam-Zeros-simple': lambda A, K, name: BigClam(1.0 * (A != 0), K, dump_name=name, stepSizeMod='simple', **model_params).fit()[0],
          #'BigClam-Mean': lambda A, K, name: BigClam(1.0 * (A < np.mean(A)), K, dump_name=name, **model_params).fit()[0],
          'BigClamWeighted': lambda A, K, name: BigClam(A, K, dump_name=name, **model_params).fit()[0],
          'BigClamWeighted-simple': lambda A, K, name: BigClam(A, K, dump_name=name, stepSizeMod='simple', **model_params).fit()[0],
          #'SparseGamma-p1': lambda A, K, name: BigClamGamma(A, K, dump_name=name, pow=1, **model_params).fit()[0],
          #'SparseGamma-p0.05': lambda A, K, name: BigClamGamma(A, K, dump_name=name, pow=0.05, **model_params).fit()[0],
          'SparseGamma': lambda A, K, name: BigClamGamma(A, K, dump_name=name, **model_params).fit()[0],
          'BigClam-orig-zeros': lambda A, K, name: bigclam_orig(1.0 * (A != 0), K),
          #'BigClamWeighted-sp10': lambda A, K, name: BigClam(A, K, dump_name=name, sparsity_coef=10,  **model_params).fit()[0],
          #'SparseGamma-sp10': lambda A, K, name: BigClamGamma(A, K, dump_name=name, sparsity_coef=10, **model_params).fit()[0],
          #'BigClam-orig-mean': lambda A, K, name: bigclam_orig(1.0 * (A < np.mean(A)), K),
          'COPRA': lambda A, K, name: copra(A, K),
          'NMF': lambda A, K, name: NMF_clust(A, K),
          #'CPM': lambda A, K, name: [list(x) for x in get_percolated_cliques(nx.from_numpy_matrix(1.0 * (A != 0)), 5)]
        }

qual_fun = {'MixedModularity': MixedModularity,
            '1-MeanConductance': lambda F, A: 1-MeanConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MeanConductance(F, A),
            '1-MaxConductance': lambda F, A: 1-MaxConductance(GetComms(F, A), A) if not isinstance(F, list) else 1-MaxConductance(F, A),
            '1-NMI': lambda F,A, true_comm: 1-NMI(GetComms(F, A), A, true_comm) if not isinstance(F, list) else 1-NMI(F, A, true_comm),
            #'NMI_new': lambda F,A, true_comm: NMI3(GetComms(F, A), A, true_comm) if not isinstance(F, list) else NMI(F, A, true_comm),
            }

# In[48]:

def enshure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

lanc_bech_files = ['outputlog', 'parameters.dat', 'community.dat', 'statistics.dat', 'time_seed.dat', 'network.dat']
nmi_files = ['outputlog-nmi', 'clu1', 'clu2']
model_files = {'COPRA': ['../external/COPRA/COPRA_output.log', '../external/COPRA/test.COPRA', 'clusters-test.COPRA'],
               'BigClam-orig-zeros': ['../external/BigClam/bigClam_output.log', '../external/BigClam/test.bigClam', '../external/BigClam/cmtyvv.txt' ]}

def worker_models(args):
    A, K, name = args
    print '!',
    F = models[name](A, K, name)
    print '.',
    return name, F
    # if name in model_files and save_all:
    #     for filename in model_files[name]:
    #         copyfile(filename, cur_save_dir + name + '-' + filename.split('/')[-1])
    #     with file(cur_save_dir + name + '-F', 'w') as f:
    #         for Fi in F:
    #             for j in Fi:
    #                 f.write('{:.4f},\t'.format(j)),
    #             f.write('\n')

def worker(iter, mix):
        res = {name: {key: 0 for key in qual_fun} for name in models}
        print ' {}:'.format(iter),
        G, comms = LancichinettiBenchmark(**data_params)
        if save_all:
            cur_save_dir = "../data/dumps/all_exp/{:.3f}/".format(mix)
            enshure_dir(cur_save_dir)
            cur_save_dir = cur_save_dir + str(iter) + '/'
            enshure_dir(cur_save_dir)
            for filename in lanc_bech_files:
                copyfile('../external/Lancichinetti benchmark/' + filename, cur_save_dir + 'testgraph-' + filename)
        A = np.array(nx.to_numpy_matrix(G))
        Fs = pool.map(worker_models, zip([A]*len(models), [len(comms)]*len(models), models.keys()))
        #Fs = [worker_models((A, len(comms), name)) for name in models]

        Fs = dict(Fs)
        # for name in models:
        #     print '!',
        #     F = models[name](A, len(comms), name)
        #     if name in model_files and save_all:
        #         for filename in model_files[name]:
        #             copyfile(filename, cur_save_dir + name + '-' + filename.split('/')[-1])
        #         with file(cur_save_dir + name + '-F', 'w') as f:
        #             for Fi in F:
        #                 for j in Fi:
        #                     f.write('{:.4f},\t'.format(j)),
        #                 f.write('\n')
        #     print '.',
        for name in Fs:
            for key in qual_fun:
                try:
                    if key not in {"1-NMI", "NMI", 'NMI_new'}:
                        q = qual_fun[key](Fs[name], A)
                    else:
                        q = qual_fun[key](Fs[name], A, comms)
                        if save_all:
                            for filename in nmi_files:
                                copyfile('../external/Lancichinetti benchmark/' + filename, cur_save_dir + name + '-' + filename)
                except:
                    print 'Some err in ' + name,
                    q = float('nan')
                res[name][key] = q
            if save_all:
                with file(cur_save_dir + name + '-' 'quals', 'w') as f:
                    for key in res[name]:
                        f.write('{}: {}\n'.format(key, res[name][key]))
        return res


def mean(l):
    return 1.0 * sum(l) / len(l)


if __name__ == '__main__':
    pool = Pool(processes=6)
    iter_count = 20
    if save_all:
        enshure_dir("../data/dumps/all_exp")
    mixing_range = np.linspace(0, 0.7, 21)
    #mixing_range = np.linspace(0, 0.5, 3)
    models_res = []
    #(models_res, mixing_range, mix, data_params) = load(file('../data/dumps/models_res_temp-3-dump'))
    for i_mix, mix in enumerate(mixing_range):
        print '{} mix: {}'.format(time(), mix)
        with file(r'..\external\Lancichinetti benchmark\time_seed.dat', 'w') as f:
            f.write(str(seed))
        data_params['on'] = np.floor(data_params['N'] * mix)
        one_graph_res = {name: {key: [] for key in qual_fun} for name in models}
        res = []
        for iter in xrange(iter_count):
            res.append(worker(iter, mix))
        #res = pool.map(worker, xrange(iter_count))
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