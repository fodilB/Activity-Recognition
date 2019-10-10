from matplotlib import pyplot as plt
import numpy as np
from sklearn import decomposition
from sklearn import cluster


def plot_clusters(X, n_clusters=10):
    pca = decomposition.PCA(n_components=2).fit(X)
    r_X = pca.transform(X)


    h = .01

    x_min, x_max = r_X[:, 0].min() + 1, r_X[:, 0].max() - 1
    y_min, y_max = r_X[:, 1].min() + 1, r_X[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    kmeans = cluster.KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(r_X)
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    # plt.figure(1)
    # plt.clf()
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    plt.plot(r_X[:, 0], r_X[:, 1], 'k.', markersize=2)

    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='.', s=169, linewidths=3, color='w', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_surface(clf, X, y,
                 xlim=(-10, 10), ylim=(-10, 10), n_steps=250,
                 subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps),
                         np.linspace(ylim[0], ylim[1], n_steps))

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.scatter(X[:, 0], X[:, 1], c=y, linewidths=0)
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    if show:
        plt.show()


def plot_histogram(clf, X, y, subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)

    if hasattr(clf, "decision_function"):
        d = clf.decision_function(X)
    else:
        d = clf.predict_proba(X)[:, 1]

    plt.hist(d[y == "b"], bins=50, normed=True, color="b", alpha=0.5)
    plt.hist(d[y == "r"], bins=50, normed=True, color="r", alpha=0.5)

    if show:
        plt.show()


def plot_clf(clf, X, y):
    plt.figure(figsize=(8, 4))
    plot_surface(clf, X, y, subplot=(1, 2, 1), show=False)
    plot_histogram(clf, X, y, subplot=(1, 2, 2), show=True)


def print_digits(images, y, max_n=10):
    # set up the figure size in inches
    print("IN")
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    i = 0
    while i < max_n and i < images.shape[0]:
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        # label the image with the target value
        p.text(0, 14, str(y[i]))
        i += 1
    plt.show()
    print("raef")




