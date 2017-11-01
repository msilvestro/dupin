"""Metrics and dissimilarity matrix."""
import numpy as np
from numba import jit
from math import gcd
from scipy.interpolate import interp1d


@jit
def dissimilarity_matrix(data, metric):
    """Compute the dissimilarity matrix of a dataset.

    Parameters
    ----------
    data : (n,) ndarray
        Data set.
    metric : function
        Function to compute pairwise distances.

    Returns
    -------
    diss : (n, n) ndarray
        A squared dissimilarity matrix, in which every item represents a
        pairwise distance between data objects.

    Notes
    -----
    Version to allow arbitrary metric as input, slower.

    """
    n = data.shape[0]
    diss = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            dist = metric(data[i], data[j])
            diss[i, j] = dist
            diss[j, i] = dist
    return diss


def _dissimilarity_matrix(metric):
    @jit(nopython=True)
    def dm(data):
        """Compute the dissimilarity matrix of a dataset.

        Parameters
        ----------
        data : (n,) ndarray
            Data set.
        metric : function
            Function to compute pairwise distances.

        Returns
        -------
        diss : (n, n) ndarray
            A squared dissimilarity matrix, in which every item represents a
            pairwise distance between data objects.

        Notes
        -----
        Version to let ``numba`` run in 'nopython' mode (faster).

        """
        n = data.shape[0]
        diss = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1):
                dist = metric(data[i], data[j])
                diss[i, j] = dist
                diss[j, i] = dist
        return diss

    return dm


def _dissimilarity_matrix_partial(metric):
    @jit(nopython=True)
    def dmp(data, start, m):
        """Compute the dissimilarity matrix of a dataset.

        Parameters
        ----------
        data : (n,) ndarray
            Data set.
        metric : function
            Function to compute pairwise distances.
        start : int
            Starting index of the given subset of the data with respect to the
            original data indices.
        m : int
            Size of the subset of the data.

        Returns
        -------
        diss : (m, n) ndarray
            A partial dissimilarity matrix, in which every item represents a
            pairwise distance between data objects.

        Notes
        -----
        Version to let ``numba`` run in 'nopython' mode (faster).

        """
        n = data.shape[0]
        diss = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                dist = metric(data[i+start], data[j])
                diss[i, j] = dist
        return diss

    return dmp


@jit(nopython=True)
def manhattan_distance(vec1, vec2):
    r"""Compute the Manhattan distance between two vectors.

    Parameters
    ----------
    vec1, vec2 : array_like
        Vectors to be used to compute the distance.

    Returns
    -------
    dist : float
        Manhattan distance between the two vectors.

    Notes
    -----
    The Manhattan distance is defined as the sum of absolute differences
    between each coordinate of the vectors, i.e. :math:`\sum_i |x_i - y_i|`.

    """
    return np.absolute(vec1 - vec2).sum()


@jit(nopython=True)
def euclidean_distance(vec1, vec2):
    r"""Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    vec1, vec2 : array_like
        Vectors to be used to compute the distance.

    Returns
    -------
    dist : float
        Euclidean distance between the two vectors.

    Notes
    -----
    The Euclidean distance is defined as the root of the sum of squared
    differences between each coordinate of the vectors, i.e.
    :math:`\sum_i (x_i - y_i)^2`.

    """
    return np.sqrt(((vec1 - vec2)**2).sum())


@jit(nopython=True)
def supremum_distance(vec1, vec2):
    r"""Compute the supremum distance between two vectors.

    Parameters
    ----------
    vec1, vec2 : array_like
        Vectors to be used to compute the distance.

    Returns
    -------
    dist : float
        Manhattan distance between the two vectors.

    Notes
    -----
    The supremum distance is defined as the maximum of absolute differences
    between each coordinate of the vectors, i.e. :math:`\max_i |x_i - y_i|`.

    """
    return np.absolute(vec1 - vec2).max()


@jit
def manhattan_warped_distance(vec1, vec2):
    r"""Compute the Manhattan distance between two vectors.
    TODO update
    TODO to be tested

    Parameters
    ----------
    vec1, vec2 : array_like
        Vectors to be used to compute the distance.

    Returns
    -------
    dist : float
        Manhattan distance between the two vectors.

    Notes
    -----
    The Manhattan distance is defined as the sum of absolute differences
    between each coordinate of the vectors, i.e. :math:`\sum_i |x_i - y_i|`.

    """
    n = vec1.shape[0]
    m = vec2.shape[0]

    def lcm(a, b):
        """Least common multiple."""
        return a * b // gcd(a, b)

    def warp(vector, length, kind='linear'):
        n = vector.shape[0]
        linspace = np.linspace(0, 1, n)
        interp = interp1d(linspace, vector, kind=kind)
        new_linspace = np.linspace(0, 1, length)
        return interp(new_linspace)

    l = lcm(n, m)
    vec1_warped = warp(vec1, l)
    vec2_warped = warp(vec2, l)
    return np.absolute(vec1_warped - vec2_warped).sum() / l
