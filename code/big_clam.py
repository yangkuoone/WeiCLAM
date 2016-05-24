from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
import cPickle
from Extends import draw_groups

#TODO: use this functions instead internal in BigClam
def conductanceLocalMin(A, K=None):
    if K is None:
        K = A.shape[1]
    InvalidNIDS = []
    NIdPhiV = GetNeighborhoodConductance(A)
    NIdPhiV = sorted(enumerate(NIdPhiV), key=lambda x: x[1])
    indx = []
    CurCID = 0
    for ui in xrange(len(NIdPhiV)):
        UID = NIdPhiV[ui][0]
        if UID in InvalidNIDS:
            continue
        indx.append(UID)
        NI = A[UID]  # neighbours of UID
        NI[UID] = 1
        NI = np.where(NI)[0]
        InvalidNIDS.extend(NI)
        CurCID += 1
        if CurCID >= K:
            break
    return indx

def GetNeighborhoodConductance(A, minDeg = 5):
    N, K = A.shape
    Edges2 = np.sum(A)
    NIdPhiV = np.zeros(shape=(N,))
    for u in xrange(N):
        GetDeg = np.sum(A[u])
        NBCmty = A[u].copy() # neighbours of u
        NBCmty[u] = 1
        NBCmty = np.where(NBCmty)

        NIdPhiV[u] = 1 if GetDeg < minDeg else GetConductance(A, NBCmty[0], Edges2)
    return NIdPhiV

#TODO: optimize
def GetConductance(A, CmtyS, Edges2):
    Vol, Cut, phi = 0, 0, 0.0
    for i in xrange(len(CmtyS)):
        NI = A[CmtyS[i]].copy()  # neighbours of u
        NI[CmtyS[i]] = 1
        NI = np.where(NI)[0]
        for e in xrange(len(NI)):
            if NI[e] not in CmtyS:
                Cut += 1
        Vol += sum(A[CmtyS[i]])

    if Vol != Edges2:
        if 2 * Vol > Edges2:
            phi = Cut / (Edges2 - Vol)
        elif Vol == 0:
            phi = 0
        else:
            phi = Cut / Vol
    elif Vol == Edges2:
        phi = 1

    return phi


class BigClam(object):
    def __init__(self, A=None, K=None, debug_output=False, LLH_output=True, sparsity_coef = 0, initF=None, eps=1e-4, iter_output=None, alpha=None):
        np.random.seed(1125582)
        self.A = A.copy()
        self.not_A = 1.0 * (self.A == 0)
        #if not selfneib:
        np.fill_diagonal(self.not_A, 0)
        self.weighted = set(np.unique(A)) != {0.0, 1.0}
        self.sparsity_coef = sparsity_coef
        self.debug_output = debug_output
        self.LLH_output = LLH_output
        self.LLH = [[], []]
        self.initFmode = initF
        self.eps = eps
        #self.epsCommProb = 1e-6
        self.epsCommForce = 1e-6
        # log (1.0 / (1.0 - PNoCom))
        self.sumF = 0
        self.maxLLH = -np.infty
        self.maxF = None
        self.hist = [[], [], [], [], [], []]
        self.noImprCount = 0
        self.K = K if K is not None else 2
        self.N = self.A.shape[0]
        self.iter_output = self.N if iter_output is None else iter_output
        self.LLH_output_vals = []
        self.alpha = alpha if alpha is not None else 0.3 if self.weighted else 0.1

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

    def gradient_check(self, F, Fu, u = None):
        D = F.copy()
        D[u] = Fu
        t = self.sumF.copy()
        self.update_sumF(D[u], F[u])
        ans = self.gradient(D, u)
        self.sumF = t
        return ans

    def draw_loglikelihood_slice(self, F, u, direction, step):
        res = []
        Fres = []
        du = F[u].copy()
        points = np.linspace(-0.5*step, 1.5*step, 1501)
        zero_indx = np.where(abs(points) < step*1e-4)[0][0]
        step_indx = np.where(abs(points-step) < step*1e-4)[0][0]
        for t in points:
            F[u] = du + t * direction
            F[u][F[u] < 0] = 0
            F[u][F[u] > 1000] = 1000
            res.append(self.loglikelihood(F))
            Fres.append(F.copy())
        ax1 = plt.subplot(311)
        plt.plot(points, res)
        plt.scatter([0.0, step], [res[zero_indx], res[step_indx]], c='r')
        plt.setp(ax1.get_xticklabels(), fontsize=6)
        ax2 = plt.subplot(312, sharex=ax1)
        Fmax = [np.max(f) for f in Fres]
        plt.plot(points, Fmax)
        plt.scatter([0.0, step], [Fmax[zero_indx], Fmax[step_indx]], c='r')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(313, sharex=ax1)
        Fzeros = [np.sum(f == 0) for f in Fres]
        plt.plot(points, Fzeros)
        plt.scatter([0.0, step], [Fzeros[zero_indx], Fzeros[step_indx]], c='r')
        plt.show()
        return res

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

    def loglikelihood(self, F, u=None, newFu=None):
        if newFu is not None:
            Fu = newFu
            sumF = self.sumF - F[u] + newFu
        else:
            Fu = F[u]
            sumF = self.sumF

        if u is None:
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
        else:
            indx = np.where(self.A[u, :] != 0)
            neigF = F[indx] / self.A[u, indx].T
            pen = self.calc_penalty(F, u, newFu)
            #if np.sum(np.abs(self.sumF-np.sum(F, axis=0))) > 1e-3:
            #    print 'sumF wrong!! sumF:{}, True:{}'.format(self.sumF, np.sum(F, axis=0))
                #self.sumF = np.sum(F, axis=0)
            return np.sum(np.log(1 - np.exp(-Fu.dot(neigF.T) - self.epsCommForce))) - Fu.dot((sumF - Fu - np.sum(neigF, axis=0).T)) - self.sparsity_coef * pen

    def gradient(self, F, u=None):
        if u is None:
            raise NotImplemented
        else:
            indx = np.where(self.A[u, :] != 0)
            neigF = F[indx] / self.A[u, indx].T
            PP = 1 / (1 - np.exp(-F[u].dot(neigF.T) - self.epsCommForce)) - 1
            pen_grad = self.calc_penalty_grad(F, u)

            grad = np.sum(neigF * PP[:, None], axis=0) - (self.sumF - F[u] - np.sum(neigF, axis=0)) - 2 * self.sparsity_coef * pen_grad

            m = max(np.abs(grad))
            if m > 100:
                grad = grad * 100.0 / m
        # is C++ code grad[grad > 10] = 10
        # is C++ code grad[grad < -10] = -10

        return grad

    def initRandF(self):
        F = 0.75 + 0.5*np.random.rand(self.N, self.K)
        self.init_sumF(F)
        self.initFmode = F.copy()
        return F

    def initZeros(self):
        F = 0.01 * np.ones((self.N, self.K))
        self.init_sumF(F)
        self.initFmode = F.copy()
        return F

    def initNeighborComF(self, A=None):
        InvalidNIDS = []
        ChosenNIDV = []
        RunTm = 0
        NIdPhiV = self.GetNeighborhoodConductance(A)
        NIdPhiV = sorted(enumerate(NIdPhiV), key=lambda x: x[1])

        CurCID = 0
        for ui in xrange(len(NIdPhiV)):
            UID = NIdPhiV[ui][0]
            if UID in InvalidNIDS:
                continue
            ChosenNIDV.append(UID) # for debug
            # add the node and its neighbors to the current community
            F[UID, CurCID] = 1.0
            NI = A[UID] # neighbours of UID
            NI[UID] = 1
            NI = np.where(NI)[0]
            for e in xrange(len(NI)):
                F[NI[e], CurCID] = 1.0
            InvalidNIDS.extend(NI)
            CurCID += 1
            if CurCID >= self.K:
                break
        else:
            if self.debug_output:
                print "{} communities needed to fill randomly\n".format(self.K - CurCID)
        sumF = np.sum(F, axis=0)
        ComSize = 10
        for i in np.where(sumF == 0)[0]:
            UID = np.random.random_integers(0, self.N-1, ComSize)
            F[UID, i] = np.random.random(size=(ComSize,))
        self.init_sumF(F)
        self.initFmode = F.copy()
        return F

    def GetNeighborhoodConductance(self, A=None):
        if A is None:
            A = self.A
        if self.weighted:
            A = self.A.copy()
            A[A < np.mean(A)] = 0
            A[A >= np.mean(A)] = 1
        F = np.zeros(shape=(self.N, self.K))
        Edges2 = np.sum(A)
        NIdPhiV = np.zeros(shape=(self.N,))
        for u in xrange(self.N):
            GetDeg = np.sum(A[u])
            NBCmty = A[u]  # neighbours of u
            NBCmty[u] = 1
            NBCmty = np.where(NBCmty)

            NIdPhiV[u] = 1 if GetDeg < 5 else self.GetConductance(A, NBCmty[0], Edges2)
        return NIdPhiV

    def GetConductance(self, A, CmtyS, Edges2):
        Vol, Cut, phi = 0, 0, 0.0
        for i in xrange(len(CmtyS)):
            NI = A[i] # neighbours of u
            NI[i] = 1
            NI = np.where(NI)[0]
            for e in xrange(len(NI)):
                if NI[e] not in CmtyS:
                    Cut += 1
            Vol += sum(A[i])

        if Vol != Edges2:
            if 2 * Vol > Edges2:
                phi = Cut / (Edges2 - Vol)
            elif Vol == 0:
                phi = 0
            else:
                phi = Cut / Vol
        elif Vol == Edges2:
            phi = 1

        return phi


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
        step = self.backtrakingLineSearch(u, F, grad, grad, alpha=self.alpha)
        if step != 0.0:
            self.hist[0].append(iter)
            self.hist[1].append(u)
            self.hist[2].append(F.copy())
            self.hist[3].append(grad.copy())
            self.hist[4].append(step)
            newFu = np.minimum(np.maximum(0, F[u] + step * grad), 10000)
            self.update_sumF(newFu, F[u])
            F[u] = newFu
            self.hist[5].append(F.copy())
        return F

    def backtrakingLineSearch(self, u, F, deltaV, gradV, alpha=0.1, beta=0.5, MaxIter=20):
        stepSize = 1 if not self.weighted else 0.1
        LLH = self.loglikelihood(F, u)
        for i in xrange(MaxIter):
            D = F[u] + stepSize * deltaV
            D[D < 0] = 0
            D[D > 10000] = 10000
            newLLH = self.loglikelihood(F, u, newFu=D)
            update = alpha * stepSize * gradV.dot(deltaV)
            if newLLH < LLH + update or np.isnan(newLLH):
                stepSize *= beta
            else:
                break
        else:
            stepSize = 0
            if self.debug_output:
                print '!',
        return stepSize

    def initFromSpecified(self):
        self.init_sumF(self.initFmode)
        return self.initFmode.copy()

    def initF(self):
        F = self.initRandF() if self.initFmode == 'rand' else self.initNeighborComF() if self.initFmode is None else \
            self.initZeros() if self.initFmode == 'zeros' else self.initFromSpecified()
        self.maxLLH = self.loglikelihood(F)
        self.LLH[0].append(0)
        self.LLH[1].append(self.maxLLH)
        self.maxF = F.copy()
        return F

    def fit(self, A=None, K=None):
        if A is not None:
            self.A = A
        if K is not None:
            self.K = K

        F = self.initF()

        for iter, u in self.nextNodeToOptimize(F):
            F = self.optimize(F, u, iter)
            if iter % self.N == 0 and self.stop(F, iter):
                if (len(self.LLH[1]) >= 2 and self.LLH[1][-1] - self.LLH[1][-2] < -1):
                    if self.debug_output:
                        print 'Warning! Big LLH decrease!'
                    # self.loglikelihood_slice(self.hist[2][-1], self.hist[1][-1], self.hist[3][-1], self.hist[4][-1])
                break
            if iter % self.iter_output == 0:
                curLLH = self.loglikelihood(F)
                self.LLH_output_vals.append(curLLH)
                if self.LLH_output:
                    print 'iter: {}, LLH:{}'.format(iter, curLLH)
                #print '    grad check: {}'.format(sum(abs(opt.check_grad(lambda x: bigClam.loglikelihood_check(F, x, w),
                #                                                         lambda x: bigClam.gradient_check(F, x, w), Q))
                #                                      for Q, w in zip([F1[q] for q in xrange(13)], range(13))))
                #print 'sumF: {1}, {0}'.format(np.sum(F), np.sum(np.abs(self.sumF - np.sum(F, axis=0))))
        return self.maxF, self.maxLLH

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
    draw_bigClam_res()
