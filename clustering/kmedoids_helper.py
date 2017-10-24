"""Helper functions for k-medoids algorithms."""
import numpy as np
from numba import jit


def _get_clusters(metric=None, method='memory'):
    # if a method requires it, check if a metric is given
    if method in ('hybrid', 'cpu') and not metric:
        print("Error: with method `{:}` a metric is necessary.")
        return

    if method == 'memory':
        return get_clusters_memory
    if method == 'hybrid':
        return lambda data, medoids: get_clusters_hybrid(data, medoids, metric)
    if method == 'cpu':
        return _get_clusters_cpu(metric)
    print("Error: method `{:}` unknown.".format(method))
    return


def _get_medoid(metric=None, method='memory'):
    # if a method requires it, check if a metric is given
    if method in ('hybrid', 'cpu') and not metric:
        print("Error: with method `{:}` a metric is necessary.")
        return

    if method == 'memory':
        return get_medoid_memory
    if method == 'hybrid':
        return _get_medoid_hybrid(metric)
    if method == 'cpu':
        return _get_medoid_cpu(metric)
    print("Error: method `{:}` unknown.".format(method))
    return


@jit
def get_clusters_memory(diss, medoids):
    r"""Compute the clusters induced by the medoids on the dissimilarity matrix.

    Parameters
    ----------
    diss : (n, n) ndarray
        Squared symmetric dissimilarity matrix.
    medoids : (n,) ndarray
        Set of medoids, given as index of data objects representing them.

    Returns
    -------
    clusterid : ndarray
        An array containing the number of the cluster to which each object was
        assigned, where the cluster number is defined as the object number of
        the objects representing the cluster centroid.
    error : float
        The within-cluster sum of distances of the clustering solution.

    Notes
    -----
    Very fast implementation. Requires enough memory to store a n\*n matrix
    (that is the dissimilarity matrix, n is the number of data objects).

    """
    # take the submatrix in which columns corresponds to the medoids, then take
    # the argmin row-wise
    clustermem = diss[:, medoids].argmin(axis=1)
    # we want a vector with medoid indices with respect to the data and not
    # positional indices, i.e. we do not want [0, 1, 2] but
    # [med_1, med_2, med_3]
    clusterid = np.empty(clustermem.shape[0], dtype=np.uint32)
    for i, medoid in enumerate(medoids):
        clusterid[clustermem == i] = medoid
    # compute also the error
    error = diss[:, medoids].min(axis=1).sum()
    return clusterid, error


@jit
def get_medoid_memory(diss, cluster):
    r"""Compute the medoid of a cluster.

    Parameters
    ----------
    diss : (n, n) ndarray
        Squared symmetric dissimilarity matrix.
    cluster : (n,) ndarray
        Set of the indices of all objects belonging to the cluster.

    Returns
    -------
    medoid : int
        Index of the object chosen as medoid of the cluster, i.e. it is the
        object that minimizes the sum of distances with respect to all the
        other cluster members.

    Notes
    -----
    Very fast implementation. Requires enough memory to store a n\*n matrix
    (that is the dissimilarity matrix, n is the number of data objects).

    """
    medoid = cluster[np.sum(
        diss[np.ix_(cluster, cluster)], axis=1
    ).argmin()]
    return medoid


@jit
def get_clusters_hybrid(data, medoids, metric):
    r"""Compute the clusters induced by the medoids on data.

    Parameters
    ----------
    data : (n,) ndarray
        Data set.
    medoids : (n,) ndarray
        Set of medoids, given as index of data objects representing them.
    metric : function
        Function to compute pairwise distances.

    Returns
    -------
    clusterid : (n,) ndarray
        An array containing the number of the cluster to which each object was
        assigned, where the cluster number is defined as the object number of
        the objects representing the cluster centroid.
    error : float
        The within-cluster sum of distances of the clustering solution.

    Notes
    -----
    Quite fast implementation. Requires enough memory to store a n\*k matrix
    (n is the number of data objects and k is the number of clusters).

    """
    # make a big matrix that in the i-th row has the distances between the i-th
    # object and the medoids
    dists = np.zeros((data.shape[0], medoids.shape[0]))
    for i, obj in enumerate(data):
        for j, med in enumerate(medoids):
            if i != med:
                dists[i, j] = metric(obj, data[med])
    # take the index corresponding to the medoid with minimum distance from the
    # object
    clustermem = dists.argmin(axis=1)
    # we want a vector with medoid indices with respect to the data and not
    # positional indices, i.e. we do not want [0, 1, 2] but
    # [med_1, med_2, med_3]
    clusterid = np.empty(clustermem.shape[0], dtype=np.uint32)
    for i, medoid in enumerate(medoids):
        clusterid[clustermem == i] = medoid
    # take the minimum row-wise and sum the resulting vector to get the error
    error = dists.min(axis=1).sum()
    return clusterid, error


def _get_medoid_hybrid(metric):
    @jit(nopython=True)
    def get_medoid_hybrid(data, cluster):
        r"""Compute the medoid of a cluster.

        Parameters
        ----------
        data : (n,) ndarray
            Data set.
        cluster : (n,) ndarray
            Set of the indices of all objects belonging to the cluster.
        metric : function
            Function to compute pairwise distances.

        Returns
        -------
        medoid : int
            Index of the object chosen as medoid of the cluster, i.e. it is the
            object that minimizes the sum of distances with respect to all the
            other cluster members.

        Notes
        -----
        Quite fast implementation. Requires enough memory to store a m\*m
        matrix (m is the size of the given cluster).

        """
        # make a dissimilarity matrix of the cluster passed in
        m = cluster.shape[0]
        diss = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1):
                dist = metric(data[cluster[i]], data[cluster[j]])
                diss[i, j] = dist
                diss[j, i] = dist
        # then take the sum by row and choose the cluster member that minimizes
        # it
        medoid = cluster[diss.sum(axis=1).argmin()]
        return medoid

    return get_medoid_hybrid


def _get_clusters_cpu(metric):
    @jit(nopython=True)
    def get_clusters_cpu(data, medoids):
        """Compute the clusters induced by the medoids on data.

        Parameters
        ----------
        data : (n,) ndarray
            Data set.
        medoids : (n,) ndarray
            Set of medoids, given as index of data objects representing them.
        metric : function
            Function to compute pairwise distances.

        Returns
        -------
        clusterid : (n,) ndarray
            An array containing the number of the cluster to which each object
            was assigned, where the cluster number is defined as the object
            number of the objects representing the cluster centroid.
        error : float
            The within-cluster sum of distances of the clustering solution.

        Notes
        -----
        Slowest implementation. Does not require to store matrices in memory.
        Version to let `numba` run in `nopython` mode (faster).

        """
        n = data.shape[0]
        k = medoids.shape[0]
        clusterid = np.empty(n, dtype=np.uint32)
        error = 0
        for i in range(n):
            # select the cluster whom medoid is closest to the current object
            min_dist = np.inf
            min_j = -1
            for j in range(k):
                if i == medoids[j]:
                    # if the object is a medoid, its cluster will not change
                    # hence end the loop
                    min_dist = 0
                    min_j = j
                    break
                else:
                    dist = metric(data[i], data[medoids[j]])
                    if dist < min_dist:
                        min_dist = dist
                        min_j = j
            clusterid[i] = medoids[min_j]
            error += min_dist

        return clusterid, error

    return get_clusters_cpu


def _get_medoid_cpu(metric):
    @jit(nopython=True)
    def get_medoid_cpu(data, cluster):
        """Compute the medoid of a cluster.

        Parameters
        ----------
        data : (n,) ndarray
            Data set.
        cluster : (n,) ndarray
            Set of the indices of all objects belonging to the cluster.
        metric : function
            Function to compute pairwise distances.

        Returns
        -------
        medoid : int
            Index of the object chosen as medoid of the cluster, i.e. it is the
            object that minimizes the sum of distances with respect to all the
            other cluster members.

        Notes
        -----
        Slowest implementation. Does not require to store matrices in memory.
        Version to let `numba` run in `nopython` mode (faster).

        """
        min_dist = np.inf
        medoid = -1
        for prop in cluster:
            # for each proposed medoid, compute the sum of distances between it
            # and each other cluster member
            dist = 0
            for j in cluster:
                if prop != j:
                    dist += metric(data[prop], data[j])

            # retain it only if it has a lower sum of distances
            if dist < min_dist:
                min_dist = dist
                medoid = prop
        return medoid

    return get_medoid_cpu
