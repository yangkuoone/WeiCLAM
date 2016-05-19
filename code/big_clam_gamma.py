import numpy as np
from big_clam import BigClam
from scipy.special import gamma, digamma
import networkx as nx
import os, cPickle
from matplotlib import pyplot as plt
from Extends import draw_groups


class BigClamGamma(BigClam):
    def __init__(self, A=None, K=None, theta=1, debug_output=True, LLH_output=True, initF=None, eps=1e-4, iter_output=None):
        super(BigClamGamma, self).__init__(A, K, debug_output=debug_output, LLH_output=LLH_output, initF=initF, eps=eps, iter_output=iter_output)
        self.theta = theta
        self.A_dump = self.A.copy()
        self.A[self.A == 0] = 0.01


    def loglikelihood(self, F, u = None, newFu=None):
        A = self.A
        if newFu is None:
            FF = F.dot(F.T) + 1
        else:
            Fu_old = F[u].copy()
            F[u] = newFu
            FF = F.dot(F.T) + 1
            F[u] = Fu_old

        S1 = np.sum(-np.log(gamma(FF)) - FF * np.log(self.theta))
        S2 = np.sum((FF - 1) * np.log(A + self.epsCommForce) - A / self.theta)
        return S1 + S2

    def gradient(self, F, u=None):
        if u is None:
            raise NotImplemented
        else:
            f = np.ma.array(F, mask=False)
            f.mask[u] = True
            DD = digamma(F[u].dot(F.T) + 1)
            #print "DD: ", np.max(DD), np.min(DD)
            #print "F: ", np.max(F), np.min(F)
            S1 = -f * DD.T[1, None]
            S2 = -f * (np.log(self.theta) - np.log(self.A[u] + self.epsCommForce))[:, None]
            #print '.'
            return np.sum(S1 + S2, axis=0)

    def initNeighborComF(self, A=None):
        if A is None:
            A = self.A.copy()

        zero = A < np.mean(A)
        A[zero] = 0
        A[np.logical_not(zero)] = 1
        res = super(BigClamGamma, self).initNeighborComF(A)
        res[res==0] = 0.001

        return res


if __name__ == "__main__":
    K = 2
    D = cPickle.load(file('../data/vk/8357895.ego'))
    G = nx.Graph(D)
    A = np.array(nx.to_numpy_matrix(G))
    N = A.shape[0]
    ids = G.node.keys()
    names = cPickle.load(file('../data/vk/8357895.frName'))
    names = dict([(x['id'], x['first_name'][0] + '.' + x['last_name']) for x in names])

    bigClamGamma = BigClamGamma(A, K)
    bigClam = BigClam(A, K)

    Fg = bigClamGamma.fit(A, K)
    F = bigClam.fit(A, K)

    mse = np.linalg.norm((A - F.dot(F.T)))
    FF_g = Fg.dot(Fg.T)
    mse_g = np.linalg.norm((A - 1/gamma(FF_g) * np.exp(-FF_g)))
    print 'MSE> exp: {}, basic: {}, win - {}'.format(mse_g, mse, 'basic' if mse < mse_g else '!!! Gamma !!!')

    plt.scatter(np.log(1e-10 + Fg[:, 0]), np.log(1e-10 + Fg[:, 1]))
    plt.show()
    draw_groups(A, F, ids, names, 'WBigClamK')

    # plt.plot(*bigClam.LLH)
    # plt.xlabel('iteration')
    # plt.ylabel('loglikelihood')
    # plt.grid(True)
    #
    # plt.show()

