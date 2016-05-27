from __future__ import division
import cPickle
import networkx as nx
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from transliterate import translit
from datetime import datetime
import sys

col = ['b', 'r', 'g', 'm', 'c', 'y', '#56A0D3', '#ED9121', '#00563F', '#062A78', '#703642', '#C95A49',
       '#92A1CF', '#ACE1AF', '#007BA7', '#2F847C', '#B2FFFF', '#4997D0',
       '#DE3163', '#EC3B83', '#007BA7', '#2A52BE', '#6D9BC3', '#007AA5',
       '#E03C31']

def toBigClamFormat(G, file_name):
    A = np.array(nx.to_numpy_matrix(G))
    G = nx.Graph(A)
    with file(file_name, 'w') as f:
        [f.write('{}\t{}\n'.format(e[0]+1, e[1]+1)) for e in G.edges()]

def fromBigClamFormat(file_name):
    data = []
    N = 0
    with file(file_name, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            a, b = line.split()
            a, b = int(a), int(b)
            data.append((a, b))
            N = max(N, a, b)
    A = np.zeros(shape=(N+1, N+1))
    for a, b in data:
        A[a][b] = 1
    return nx.Graph(A)


def fromBigClamFormat_sparse(file_name):
    data = []
    with file(file_name, 'r') as f:
        data = [map(int, line.split()) for line in f if line[0] != '#']
    G = nx.Graph(data=data)
    return G


def test_example():
    return np.array([[0, 1, 1, 1, 0, 0, 0],
                     [1, 0, 1, 1, 0, 0, 0],
                     [1, 1, 0, 1, 0, 0, 0],
                     [1, 1, 1, 0, 1, 1, 1],
                     [0, 0, 0, 1, 0, 1, 1],
                     [0, 0, 0, 1, 1, 0, 1],
                     [0, 0, 0, 1, 1, 1, 0]])


def draw_groups(A, F, ids, names, figname = 'NoName', png=True, pdf=False, display=False, svg=False, dpi=2300):
    N, K = F.shape

    C = F > np.sum(A) / (A.shape[0] * (A.shape[0] - 1))
    indx = np.argmax(F, axis=1)
    for i in xrange(N):
        C[i, indx[i]] = True
    print F
    print C

    comm = [[] for i in xrange(N)]
    for x, y in zip(*np.where(C)):
        comm[x].append(y)
    u_comm = np.unique(comm)

    comm2id = []
    for u in u_comm:
        comm2id.append([i for i, c in enumerate(comm) if c == u])

    G = nx.Graph(A)
    plt.figure(num=None, figsize=(10, 10))

    pos = []
    centers = [np.array([0, 1])]
    angle = np.pi / K
    turn = np.array([[np.cos(2*angle), np.sin(2*angle)], [-np.sin(2*angle), np.cos(2*angle)]])
    radius = np.sin(angle)
    new_pos = {i: [] for i in xrange(N)}

    U, s, V = np.linalg.svd(F.T.dot(F))
    posSVD =[x[0] for x in sorted([x for x in enumerate(U[0])], key= lambda x: x[1])]

    for i in xrange(K):
        if i + 1 != K:
            centers.append(turn.dot(centers[-1]))

    for i in xrange(K):
        for key, value in nx.spring_layout(G.subgraph(np.where(C[:, posSVD[i]])[0])).iteritems():# positions for all nodes
            new_pos[key].append(value * radius + 0.8 * centers[posSVD[i]])

    for key in new_pos:
        new_pos[key] = np.sum(np.array(new_pos[key]), axis=0) / (1.5 * len(new_pos[key])) ** 1.2

    for val in comm2id:
        if len(comm[val[0]]) < 2:
            continue
        m = np.mean(np.array([new_pos[x] for x in val]), axis=0)
        for x in val:
            new_pos[x] = 0.8 * len(comm[val[0]]) * (new_pos[x] - m) + m

    nx.draw_networkx_edges(G, new_pos, width=0.25, alpha=0.07)
    nx.draw_networkx_nodes(G, new_pos, node_color='#BBBBBB', node_size=15, alpha=1, linewidths=0)
    for j in xrange(C.shape[0]):
        k = 0
        for i in xrange(C.shape[1]):
            if(C[j][i]):
                nx.draw_networkx_nodes(G, new_pos, nodelist=[j], node_color=col[i], node_size=10-1*k,
                                       alpha=0.6, linewidths=0)
                k += 1
    labels = {i: u' '.join([str(n) for n in np.where(c)[0]]) + u'\n> {} <'.format(translit(names[ids[i]].replace(u'\u0456', u'~'), u'ru', reversed=True)) for i, c in enumerate(C)}
    nx.draw_networkx_labels(G, new_pos, labels, font_size=0.1)
    plt.axis('off')

    if pdf:
        plt.savefig("../plots/{}.pdf".format(figname))
    if png:
        plt.savefig("../plots/{}.png".format(figname), dpi=dpi)
    if svg:
        plt.savefig("../plots/{}.svg".format(figname))
    if display:
        plt.show() # display

def progress(list, update_interval=1):

    """
    display progress for loop list
    :param list: list
    :param update_interval: minimal update iterval for progress
    :return: generator with progress output to stdout
    """
    N = len(list)
    start = datetime.now()
    last = start
    for index, val in enumerate(list):
        yield val
        time = datetime.now()
        if (time - last).seconds > update_interval:
            sys.stdout.write("\rProgress: {:.2f}% | ETA/Total: {:.2f}/{:.2f} sec {}"
                             .format(100.0 * (index + 1) / N, (time - start).seconds / (index + 1) * (N - 1 - index),
                                     (time - start).seconds / (index + 1) * N, " " * 30))
            sys.stdout.flush()
            last = time


if __name__ == '__main__':
    D = cPickle.load(file('../data/vk/3771369.ego'))
    G = nx.Graph(D)
    toBigClamFormat(G, '../data/vk/3771369.bigClam')
    G = fromBigClamFormat('../data/test.bigclam')
    A = np.array(nx.to_numpy_matrix(G))
    pass

def conductance(G, S, T=None, weight=None):
    """Returns the conductance of two sets of nodes.
    Fixed by Zurk

    The *conductance* is the quotient of the cut size and the smaller of
    the volumes of the two sets. [1]

    Parameters
    ----------
    G : NetworkX graph

    S : sequence
        A sequence of nodes in `G`.

    T : sequence
        A sequence of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The conductance between the two sets `S` and `T`.

    See also
    --------
    cut_size
    edge_expansion
    normalized_cut_size
    volume

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    if T is None:
        T = set(G) - set(S)
    num_cut_edges = nx.algorithms.cuts.cut_size(G, S, T, weight=weight)
    volume_S = nx.algorithms.cuts.volume(G, S, weight=weight)
    volume_T = nx.algorithms.cuts.volume(G, T, weight=weight)

    if volume_S == 0:
        return 0
    if volume_T == 0:
        return 1

    return num_cut_edges / min(volume_S, volume_T)

def getSeedCenters(A, K=None, w=1, pool=None):
    G = nx.Graph(A)
    if K is None:
        K = nx.number_of_nodes(G)
    cond = GetNeighborhoodConductance(G, pool=pool)
    local_max = conductanceLocalMin(G, K, cond, pool)
    res = [local_max.pop(0)]

    while len(res) < K and len(local_max) > 0:
        all_nodes = np.any(A[res] != 0, axis=0)
        interseption = A[local_max].dot(all_nodes.T) / np.sum(A[local_max], axis=1)
        k = np.argmin(cond[local_max] + w * interseption)
        res.append(local_max.pop(k))
    return res

def conductanceLocalMin(G, K=None, cond=None, pool=None):
    if isinstance(G, np.ndarray):
        G = nx.Graph(G)
    if K is None:
        K = nx.number_of_nodes(G)
    InvalidNIDS = []
    if cond is None:
        cond = GetNeighborhoodConductance(G, pool=pool)
    cond = sorted(enumerate(cond), key=lambda x: x[1])
    indx = []
    CurCID = 0
    for ui in xrange(len(cond)):
        UID = cond[ui][0]
        if UID in InvalidNIDS:
            continue
        indx.append(UID)
        NI = G[UID]  # neighbours of UID
        NI = np.where(NI)[0]
        InvalidNIDS.extend(NI)
        InvalidNIDS.append(UID)
        CurCID += 1
        if CurCID >= K:
            break
    return indx

def getEgoGraphNodes(G, u):
    return [u] + [x for x in G.neighbors(u)]


def GetNeighborhoodConductance_worker(args):
    u, G = args[0], args[1]
    minDeg = 10
    return 1 if G.degree(u, weight='weight') < minDeg else conductance(G, getEgoGraphNodes(G, u))

def GetNeighborhoodConductance(G, minDeg = 10, pool=None):
    N = len(G)
    Edges2 = 2*len(G.edge)
    if pool is not None:
        NIdPhiV = np.array(pool.map(GetNeighborhoodConductance_worker, ((u, G) for u in G)))
    else:
        NIdPhiV = np.zeros(shape=(N,))
        for u in progress(G):
            NIdPhiV[u] = 1 if G.degree(u, weight='weight') < minDeg else conductance(G, getEgoGraphNodes(G, u))
    return NIdPhiV


def draw_loglikelihood_slice(bigClam, F, u, direction, step):
    res = []
    Fres = []
    du = F[u].copy()
    points = np.linspace(-0.5 * step, 1.5 * step, 1501)
    zero_indx = np.where(abs(points) < step * 1e-4)[0][0]
    step_indx = np.where(abs(points - step) < step * 1e-4)[0][0]
    for t in points:
        F[u] = du + t * direction
        F[u][F[u] < 0] = 0
        F[u][F[u] > 1000] = 1000
        res.append(bigClam.loglikelihood(F))
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

def gradient_check(bigClam, F, Fu, u=None):
    D = F.copy()
    D[u] = Fu
    t = bigClam.sumF.copy()
    bigClam.update_sumF(D[u], F[u])
    ans = bigClam.gradient(D, u)
    bigClam.sumF = t
    return ans
