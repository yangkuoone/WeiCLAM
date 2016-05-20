import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def log_progress(sequence, every=None, size=None):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = size / 200  # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{index} / ?'.format(index=index)
                else:
                    progress.value = index
                    label.value = u'{index} / {size}'.format(
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = str(index or '?')


test_size = 1
Fs2 = [2 * np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]]),
       2 * np.array([[1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1]]),
       2 * np.array([[1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]]),
       2 * np.array([[1, 1, 1, 5, 5, 5, 0, 0], [0, 0, 5, 5, 5, 1, 1, 1]]),
       1 * np.array([[4, 4, 4, 4, 4, 1, 1, 1], [1, 1, 1, 4, 4, 4, 4, 4]]),
       1 * np.array([[3, 3, 3, 5, 5, 5, 1, 1], [1, 1, 5, 5, 5, 3, 3, 3]]),
       2 * np.array([[1] * 100 * test_size + [0] * 100 * test_size, [0] * 100 * test_size + [1] * 100 * test_size]),
       2 * np.array([[1] * 150 * test_size + [0] * 50 * test_size, [0] * 50 * test_size + [1] * 150 * test_size]),
       2 * np.array([[1] * 120 * test_size + [0] * 80 * test_size, [0] * 80 * test_size + [1] * 120 * test_size]), ]

Fs3 = [2 * np.array([[1] * 80 * test_size + [0] * 60 * test_size,
                     [0] * 20 * test_size + [1] * 20 * test_size + [0] * 20 * test_size + [1] * 20 * test_size +
                     [0] * 20 * test_size + [1] * 20 * test_size + [1] * 20 * test_size,

                     [0] * 40 * test_size + [1] * 40 * test_size + [1] * 20 * test_size +
                     [1] * 20 * test_size + [0] * 20 * test_size]), ]


# generate agency matix from F matrix for gamma model
def gamma_model_test_data(F=None):
    np.random.seed(1122)
    if F is None:
        F = 2 * np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
    theta = F.T.dot(F)
    A = np.zeros(theta.shape)
    for i in xrange(theta.shape[0]):
        for j in xrange(theta.shape[1]):
            A[i][j] = np.random.gamma(1 + theta[i][j], 1)
    np.fill_diagonal(A, 0)
    return (A + A.T) / 2.0


# generate agency matix from F matrix
def big_clam_model_test_data(F=None):
    np.random.seed(1122)
    if F is None:
        F = 2 * np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
    theta = F.T.dot(F)
    P = 1 - np.exp(-theta)
    A = np.random.rand(*P.shape) - 1e-2 < P
    np.fill_diagonal(A, 0)
    A = (A + A.T) / 2.0
    res = 1.0 * (A != 0)
    return res


def draw_matrix(photo_l, title="", hide_ticks=True):
    if photo_l.shape[0] / photo_l.shape[1] > 5 or photo_l.shape[0] / photo_l.shape[1] < 0.2:
        k = np.floor(np.max(photo_l.shape) / np.min(photo_l.shape) / 2)
        if photo_l.shape[1] < photo_l.shape[0]:
            photo_l = np.reshape(np.tile(photo_l.copy(), (k, 1)).T, (photo_l.shape[1] * k, photo_l.shape[0])).T
        else:
            photo_l = np.reshape(np.tile(photo_l.copy(), (1, k)), (photo_l.shape[0] * k, photo_l.shape[1]))
    ax = plt.gca()
    im = plt.imshow(photo_l, interpolation='none')
    if hide_ticks:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.title(title, y=1.02, x=0.6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.07)
    plt.colorbar(im, cax=cax, ticks=np.linspace(np.min(photo_l), np.max(photo_l), 5))


def draw_res(B, F_true, F_model, F_model7=None):
    Xs, Ys = 2, 5
    f = plt.figure(figsize=(18, 6))

    plt.subplot(Xs, Ys, 1)
    draw_matrix(B, 'Agency matix sample (B)')
    plt.subplot(Xs, Ys, 2)
    draw_matrix(F_model.dot(F_model.T), 'Reconstructed matix ({} comm)'.format(F_model.shape[1]))
    plt.subplot(Xs, Ys, 3)
    draw_matrix(np.abs(F_model.dot(F_model.T) - F_true.T.dot(F_true)), 'diff ({} comm)'.format(F_model.shape[1]))

    plt.subplot(Xs, Ys, 6)
    draw_matrix(F_true, "true F value")
    plt.subplot(Xs, Ys, 7)
    draw_matrix(F_model.T, "Reconstructed F ({} comm)".format(F_model.shape[1]))
    C = F_model > np.mean(F_model) * 0.9
    plt.subplot(Xs, Ys, 8)
    draw_matrix(C.T, "Membership ({} comm)".format(F_model.shape[1]))

    if F_model7 is not None:
        plt.subplot(Xs, Ys, 4)
        draw_matrix(F_model7.dot(F_model7.T), 'Reconstructed matix ({} comm)'.format(F_model7.shape[1]))
        plt.subplot(Xs, Ys, 5)
        draw_matrix(np.abs(F_model7.dot(F_model7.T) - F_true.T.dot(F_true)), 'diff ({} comm)'.format(F_model7.shape[1]))
        plt.subplot(Xs, Ys, 9)
        draw_matrix(F_model7.T, "Reconstructed F ({} comm)".format(F_model7.shape[1]))
        C7 = F_model7 > np.mean(F_model7) * 0.5
        plt.subplot(Xs, Ys, 10)
        draw_matrix(C7.T, "Membership ({} comm)".format(F_model7.shape[1]))

        # plt.subplot(Xs, Ys, 9)
        # draw_matrix(np.abs(F_model.T - F_true), "diff (3 comm)")


def draw_test_sample(F, A, B=None, C=None, x=0):
    Ys = 5 if C is not None else 4 if B is not None else 3
    plt.figure(figsize=(3+Ys*3,6))
    plt.subplot(1, Ys, 1)
    draw_matrix(F, "True F value")
    plt.subplot(1, Ys, 2)
    draw_matrix(F.T.dot(F), "A generation model")
    plt.subplot(1, Ys, 3)
    draw_matrix(A, "Agency matix sample (A)")
    if B is not None:
        plt.subplot(1, 5, 4)
        draw_matrix(B, "Agency matix sparse sample (B)".format(x))
    if C is not None:
        plt.subplot(1, 5, 5)
        draw_matrix(C, "binary sparse sample (C)".format(x))
