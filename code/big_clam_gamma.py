import numpy as np
from big_clam import BigClam
from scipy.special import gamma, digamma, gammaln
import networkx as nx
import os, cPickle
from matplotlib import pyplot as plt
from Extends import draw_groups
from Experiments import draw_matrix


class BigClamGamma(BigClam):
    def __init__(self, A=None, K=None, theta=1, debug_output=False, LLH_output=True, sparsity_coef = 0, initF='cond', eps=1e-4,
                 iter_output=None, alpha=None, rand_init_coef=0.1, stepSizeMod="simple", processesNo=None, save_hist=False, pow=0.2, max_iter=1000000, dump=False, dump_name = None):
        self.weights = A.copy()
        A = 1.0 * (self.weights != 0)

        super(BigClamGamma, self).__init__(A, K, debug_output, LLH_output, sparsity_coef, initF, eps,
                 iter_output, alpha, rand_init_coef, stepSizeMod, processesNo, save_hist, max_iter, dump, dump_name)
        self.theta = theta
        self.Fbord = 1
        self.logA = np.log(self.weights + self.epsCommForce)
        self.sqrt_pow = np.sqrt(pow)

    def initRandF(self):
        F = 0 + np.sqrt(np.max(self.weights))*np.random.rand(self.N, self.K)
        return F

    def loglikelihood_u(self, F, u=None, newFu=None):
        llh_u = super(BigClamGamma, self).loglikelihood_u(self.sqrt_pow * F, u, newFu)

        if newFu is not None:
            Fu = newFu
        else:
            Fu = F[u]

        indx = self.A[u, :] != 0
        neigF = F[indx, :]
        FF = Fu.dot(neigF.T)
        S1 = np.sum(-gammaln(FF + self.Fbord) - (FF + self.Fbord) * np.log(self.theta))
        S2 = np.sum(FF * self.logA[indx, u] - self.weights[indx, u] / self.theta)

        return llh_u + S1 + S2

    def loglikelihood_w(self, F):
        FF = F.dot(F.T)
        P = -gammaln(FF + self.Fbord) - (FF + self.Fbord) * np.log(self.theta)
        S1 = np.sum(P[self.A == 1])
        P2 = (FF * self.logA - self.weights / self.theta)
        S2 = np.sum(P2[self.A == 1])
        return S1 + S2

    def loglikelihood(self, F):
        llh = super(BigClamGamma, self).loglikelihood(self.sqrt_pow * F)
        llh_w = self.loglikelihood_w(F)
        return llh + llh_w

    def gradient(self, F, u=None):
        grad = super(BigClamGamma, self).gradient(self.sqrt_pow * F, u)
        grad_w = self.gradient_w(F, u)
        res = grad + grad_w
        m = max(np.abs(res))
        if m > 100:
            res = res * 100.0 / m
        return res

    def gradient_w(self, F, u=None):
        if u is None:
            raise NotImplemented
        else:
            FF = F[u].dot(F.T)
            DD = digamma(FF + self.Fbord)
            S1 = DD.T[self.A[:, u] == 1, None]
            S2 = (np.log(self.theta) - self.logA[self.A[:, u] == 1, u])[:, None]
            f = -F[self.A[:, u] == 1, :]
            grad = np.sum(f * (S1 + S2), axis=0)

            return grad


if __name__ == "__main__":
    # K = 2
    # D = cPickle.load(file('../data/vk/8357895.ego'))
    # G = nx.Graph(D)
    # A = np.array(nx.to_numpy_matrix(G))
    # N = A.shape[0]
    # ids = G.node.keys()
    # names = cPickle.load(file('../data/vk/8357895.frName'))
    # names = dict([(x['id'], x['first_name'][0] + '.' + x['last_name']) for x in names])
    #
    # bigClamGamma = BigClamGamma(A, K)
    # bigClam = BigClam(A, K)
    #
    # Fg = bigClamGamma.fit(A, K)
    # F = bigClam.fit(A, K)
    #
    # mse = np.linalg.norm((A - F.dot(F.T)))
    # FF_g = Fg.dot(Fg.T)
    # mse_g = np.linalg.norm((A - 1/gamma(FF_g) * np.exp(-FF_g)))
    # print 'MSE> exp: {}, basic: {}, win - {}'.format(mse_g, mse, 'basic' if mse < mse_g else '!!! Gamma !!!')
    #
    # plt.scatter(np.log(1e-10 + Fg[:, 0]), np.log(1e-10 + Fg[:, 1]))
    # plt.show()
    # draw_groups(A, F, ids, names, 'WBigClamK')

    from Experiments import *

    F_true = Fs3[0]
    A = gamma_model_test_data(F_true)
    power = 0.05
    P = 1 - np.exp(- power * A)
    rand = np.random.rand(*A.shape)
    mask = P <= (rand + rand.T) / 2

    B = A.copy()
    B[mask] = 0
    C = B.copy()
    C[B != 0] = 1

    w_model3r = BigClamGamma(B, 3, debug_output=False, LLH_output=True, initF='cond_new_randz_spr', iter_output=1, processesNo=1,
                             rand_init_coef=0.5, stepSizeMod="backtracking0")
    F_model3r, LLH3r = w_model3r.fit()

    draw_res(B, F_true, F_model3r)
    plt.show()
    pass

    # plt.plot(*bigClam.LLH)
    # plt.xlabel('iteration')
    # plt.ylabel('loglikelihood')
    # plt.grid(True)
    #
    # plt.show()

