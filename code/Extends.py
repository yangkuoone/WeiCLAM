import cPickle
import networkx as nx
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from transliterate import translit

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

if __name__ == '__main__':
    D = cPickle.load(file('../data/vk/3771369.ego'))
    G = nx.Graph(D)
    toBigClamFormat(G, '../data/vk/3771369.bigClam')
    G = fromBigClamFormat('../data/test.bigclam')
    A = np.array(nx.to_numpy_matrix(G))
    pass