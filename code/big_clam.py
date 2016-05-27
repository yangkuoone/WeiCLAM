from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import cPickle
from Extends import draw_groups, getSeedCenters, conductanceLocalMin, progress
from multiprocessing import Pool, Process


class BigClam(object):
    def __init__(self, A=None, K=None, debug_output=False, LLH_output=True, sparsity_coef = 0, initF='cond', eps=1e-4,
                 iter_output=None, alpha=None, rand_init_coef=0.1, stepSizeMod="backtracking", processesNo=None, save_hist=False):
        np.random.seed(1125582)
        self.A = A.copy()
        self.K = K
        self.N = self.A.shape[0]
        self.not_A = 1.0 * (self.A == 0)
        np.fill_diagonal(self.not_A, 0)
        self.weighted = set(np.unique(A)) != {0.0, 1.0}
        self.sparsity_coef = sparsity_coef
        self.debug_output = debug_output
        self.LLH_output = LLH_output
        self.initFmode = initF
        self.eps = eps
        self.epsCommForce = 1e-6

        self.iter_output = self.N if iter_output is None else iter_output
        self.alpha = alpha if alpha is not None else 0.3 if self.weighted else 0.1
        self.rand_init_coef = rand_init_coef
        if processesNo != 1:
            self.pool = Pool(processesNo)
        else:
            self.pool = None
        self.save_hist = save_hist
        self.stepSizeMod = stepSizeMod


    def init(self):
        if self.save_hist:
            self.hist = [[], [], [], [], []]
        self.LLH = [[], []]
        self.noImprCount = 0
        self.LLH_output_vals = []
        self.NIdPhiV = None

        return self.initF()

    def init_sumF(self, F):
        self.sumF = np.sum(F, axis=0)

    def update_sumF(self, newFu, oldFu):
        self.sumF -= oldFu
        self.sumF += newFu

    def loglikelihood_check(self, F, Fu, u):
        D = F.copy()
        D[u] = Fu
        t = self.sumF.copy()
        self.update_sumF(D[u], F[u])
        ans = self.loglikelihood(D, u)
        self.sumF = t
        return ans


    def calc_penalty(self, F, u=None, newFu=None, real=False):
        if newFu is not None:
            Fu = newFu[:, None]
        else:
            Fu = F[u, None]

        pen = 0
        if self.sparsity_coef != 0 or real:
            if u is None:
                pen = F.T.dot(F)
                # print pen
                np.fill_diagonal(pen, 0)
                pen = np.sum(pen)
            else:
                pen = Fu.T.dot(Fu)
                np.fill_diagonal(pen, 0)
                pen = np.sum(pen)
        #print pen
        return pen

    def calc_penalty_grad(self, F, u):
        pen_grad = 0
        if self.sparsity_coef != 0:
            pen_grad = np.sum(F[u]) - F[u]
        return pen_grad

    def loglikelihood_u(self, F, u=None, newFu=None):
        if newFu is not None:
            Fu = newFu
            sumF = self.sumF - F[u] + newFu
        else:
            Fu = F[u]
            sumF = self.sumF

        indx = self.A[u, :] != 0
        neigF = F[indx, :] / self.A[u, indx].T[:, None] if self.weighted else F[indx]
        pen = self.calc_penalty(F, u, newFu)
        S1 = np.sum(np.log(1 - np.exp(-Fu.dot(neigF.T) - self.epsCommForce)))
        S2 = - Fu.dot((sumF - Fu - np.sum(neigF, axis=0).T))

        return S1 + S2 - self.sparsity_coef * pen

    def loglikelihood(self, F):
        FF = F.dot(F.T)
        if not self.weighted:
            P = np.log(1 - np.exp(-FF - self.epsCommForce))
            pen = self.calc_penalty(F)
            llh = np.sum(P * self.A) - np.sum(self.not_A * FF)
            return llh - self.sparsity_coef * pen
        else:
            indx = np.where(self.A != 0)
            P = np.log(1 - np.exp(-FF[indx] / self.A[indx] - self.epsCommForce))
            llh = np.sum(P) - np.sum(FF[self.not_A == 1])
            pen = self.calc_penalty(F)
            return llh - self.sparsity_coef * pen


    def gradient(self, F, u=None):
        if u is None:
            raise NotImplemented
        else:
            indx = np.where(self.A[u, :] != 0)
            neigF = F[indx] / self.A[u, indx].T
            PP = 1 / (1 - np.exp(-F[u].dot(neigF.T) - self.epsCommForce)) - 1
            pen_grad = self.calc_penalty_grad(F, u)

            grad = np.sum(neigF * PP[:, None], axis=0) - (self.sumF - F[u] - np.sum(neigF, axis=0)) - 2 * self.sparsity_coef * pen_grad

        return grad

    def initRandF(self):
        F = 0.75 + 0.5*np.random.rand(self.N, self.K)
        return F

    def initNeighborComF(self, A=None, new=False, randz=False, spreading=False):
        if A is None:
            A = self.A

        if self.NIdPhiV is None:
            NIdPhiV = getSeedCenters(A, self.K, pool=self.pool) if new else conductanceLocalMin(A, self.K, pool=self.pool)
            self.NIdPhiV = NIdPhiV
        else:
            NIdPhiV = self.NIdPhiV


        F = np.zeros((self.N, self.K))
        F[:, 0:len(NIdPhiV)] = A[NIdPhiV].T

        if len(NIdPhiV) < self.K:
            if self.debug_output:
                print "{} communities needed to fill randomly\n".format(self.K - len(NIdPhiV))
            ComSize = 10
            for i in xrange(len(NIdPhiV)+1, self.K):
                UID = np.random.random_integers(0, self.N-1, ComSize)
                F[UID, i] = np.random.random(size=(ComSize,))
        if spreading:
            for i in xrange(F.shape[1]):
                indx = np.any(A[F[:, i] != 0], axis=0)
                F[(indx & (F[:, i] == 0)), i] = 0.5
        if randz:
            eps = 0
            s = np.sum(F == 0)
            F[F==0] = self.rand_init_coef * np.random.random(size=(s,)) + eps


        return F

    def stop(self, F, iter):
        newLLH = self.loglikelihood(F)
        if newLLH > self.maxLLH:
            self.maxLLH = newLLH
            self.noImprCount = 0
            self.maxF = F.copy()
        else:
            self.noImprCount += 1
        self.LLH[0].append(iter)
        self.LLH[1].append(newLLH)
        return False if len(self.LLH[0]) <= 1 else abs(self.LLH[1][-1] / self.LLH[1][-2] - 1) < self.eps or self.noImprCount > 3
        #return False if len(self.LLH[0]) <= 1 else self.LLH[1][-1] - self.LLH[1][-2] <= 1e-6 * abs(self.LLH[1][-2]) and self.LLH[1][-1] != self.LLH[1][-2] and len(self.LLH[1]) > 5000

    def nextNodeToOptimize(self, F):
        iter = 0
        while True:
            order = np.random.permutation(self.A.shape[0])
            for i in order:
                for j in xrange(1):
                    iter += 1
                    yield iter, i

    def optimize(self, F, u, iter=1, step=None):
        grad = self.gradient(F, u)
        m = max(np.abs(grad))
        if m > 100:
           grad = grad * 100.0 / m
        # is C++ code grad[grad > 10] = 10
        # is C++ code grad[grad < -10] = -10
        step = self.stepSize(u, F, grad, grad, iter, alpha=self.alpha)
        if step != 0.0:
            if self.save_hist:
                self.hist[0].append(iter)
                self.hist[1].append(u)
                self.hist[2].append(F.copy())
                self.hist[3].append(grad.copy())
                self.hist[4].append(step)
            newFu = self.step(F[u], step, grad)
            self.update_sumF(newFu, F[u])
            F[u] = newFu
        return F

    def step(self, Fu, stepSize, direction):
        return np.minimum(np.maximum(0, Fu + stepSize * direction), 10000)

    def stepSize(self, u, F, deltaV, gradV, iter, alpha=0.1, beta=0.3, MaxIter=15):
        return self.backtrackingLineSearch(u, F, deltaV, gradV, alpha, beta, MaxIter) if self.stepSizeMod == 'backtracking' else 0.01 / iter ** 0.25

    def backtrackingLineSearch(self, u, F, deltaV, gradV, alpha=0.1, beta=0.3, MaxIter=15):
        stepSize = 0.1 if not self.weighted else 0.1
        LLH = self.loglikelihood_u(F, u)
        for i in xrange(MaxIter):
            D = self.step(F[u], stepSize, deltaV)
            newLLH = self.loglikelihood_u(F, u, newFu=D)
            update = alpha * stepSize * gradV.dot(deltaV)
            if newLLH < LLH + update or np.isnan(newLLH):
                stepSize *= beta
            else:
                break
        else:
            stepSize = 0
        return stepSize

    def initFromSpecified(self):
        return self.initFmode.copy()

    def initF(self):
        inits = {
            'cond': self.initNeighborComF,
            'cond_new': lambda: self.initNeighborComF(new=True),
            'cond_randz': lambda: self.initNeighborComF(new=False, randz=True),
            'cond_new_randz': lambda: self.initNeighborComF(new=True, randz=True),
            'cond_randz_spr': lambda: self.initNeighborComF(new=False, randz=True, spreading=True),
            'cond_new_randz_spr': lambda: self.initNeighborComF(new=True, randz=True, spreading=True),
            'rand': self.initRandF,
            'def': self.initFromSpecified
        }

        try:
            F = inits['def']() if not isinstance(self.initFmode, basestring) else inits[self.initFmode]()
        except (KeyError):
            print 'No such init mode: \'{}\' use from this list:{} or set F manualy'.format(self.initFmode, ', '.join(inits))
            raise

        self.maxLLH = self.loglikelihood(F)
        self.LLH[0].append(0)
        self.LLH[1].append(self.maxLLH)
        self.maxF = F.copy()
        self.initFmode = F.copy()
        self.init_sumF(F)
        return F

    def fit_known_k(self, K):
        self.K = K
        F = self.init()


        for iter, u in self.nextNodeToOptimize(F):
            F = self.optimize(F, u, iter)
            if iter % self.N == 0 and self.stop(F, iter):
                if (len(self.LLH[1]) >= 2 and self.LLH[1][-1] - self.LLH[1][-2] < -1):
                    if self.debug_output:
                        print 'Warning! Big LLH decrease!'
                break
            if iter % self.iter_output == 0:
                curLLH = self.loglikelihood(F)
                self.LLH_output_vals.append(curLLH)
                if self.LLH_output:
                    print 'iter: {}, LLH:{}'.format(iter, curLLH)
        return self.maxF, self.maxLLH

    def fit_unknown_k(self):
        bord = 0.005
        K = 2
        res = []
        while K < 10000:
            if self.debug_output:
                print '{} communities...'.format(K)
            res.append(self.fit_known_k(K))
            if len(res) > 1:
                if (res[-1][1] - res[-2][1]) / res[0][1] < - bord:
                    break
            K += 1

        return res[-1]

    def fit(self, K=None):
        if K is not None:
            self.K = K
        if self.K is not None:
           return self.fit_known_k(self.K)
        else:
            return self.fit_unknown_k()



def draw_bigClam_res():
    K = 9
    D = cPickle.load(file('../data/vk/8357895.ego'))
    G = nx.Graph(D)
    A = np.array(nx.to_numpy_matrix(G))
    N = A.shape[0]
    ids = G.node.keys()
    names = cPickle.load(file('../data/vk/8357895.frName'))
    names = dict([(x['id'], x['first_name'][0] + '.' + x['last_name']) for x in names])
    print np.sum(A) / (A.shape[0] * (A.shape[0] - 1))
    sparsity_coef = 1
    bigClam = BigClam(A, K, sparsity_coef=sparsity_coef)

    F = bigClam.fit(A, K)

    draw_groups(A, F, ids, names, 'BigClamK_{}sp'.format(sparsity_coef))

    # plt.plot(*bigClam.LLH)
    # plt.xlabel('iteration')
    # plt.ylabel('loglikelihood')
    # plt.grid(True)
    #
    # plt.show()

if __name__ == "__main__":
    #draw_bigClam_res()
    #A = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1],
                  #[0, 0, 0, 1, 1, 0]])
    #print GetNeighborhoodConductance(A)


    from Experiments import *
    F_true = Fs3[0]
    A = gamma_model_test_data(F_true)
    power = 0.2
    P = 1 - np.exp(- power * A)
    mask = P <= np.random.rand(*A.shape)

    B = A.copy()
    B[mask] = 0
    C = B.copy()
    C[B != 0] = 1

    # eps = 1e-6
    # comm_count = range(3, 10)
    # init = None
    # Fs = []
    # LLHs = []
    # for i in progress(comm_count):
    #     bc = BigClam(C, i, initF='rand', debug_output=False, LLH_output=True, eps=eps, processesNo=4)
    #     res = bc.fit()
    #     Fs.append(res[0])
    #     LLHs.append(res[1])

    # import networkx as nx
    #
    # DATA_PATH = r'../data/SNAP/facebook_combined.txt'
    # G = nx.read_edgelist(DATA_PATH)
    # A = nx.adjacency_matrix(G).toarray()
    # bc = BigClam(A, initF='cond_new_randz')
    # F = bc.fit()