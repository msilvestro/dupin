"""Various implementations of the k-medoids algorithm."""
import numpy as np
from numba import jit
from clustering.kmedoids_helper import _get_clusters, _get_medoid, compute_error
from clustering.metrics import _dissimilarity_matrix


def pam(data, k, metric=None, method='memory'):
    """Partitioning Around Medoids, a realization of k-medoids.

    Parameters
    ----------
    data : (n,) ndarray or (n, n) ndarray
        Data set or dissimilarity matrix.
    k : int
        Number of desired clusters.
    metric : function, optional
        Function to compute pairwise distances.
    method : {'memory', 'hybrid', 'cpu'}
        Implementation to use.

    Returns
    -------
    clusterid : (n,) ndarray
        An array containing the number of the cluster to which each object was
        assigned, where the cluster number is defined as the object number of
        the objects representing the cluster centroid.
    error : float
        The within-cluster sum of distances of the clustering solution.

    Algorithm
    ---------
    1. Arbitrarily choose k data objects as the initial medoids.
    2. Assign every data object to its closest medoid.
    3. Recompute the medoid for every cluster as the member of the
       cluster that minimizes the sum of distances with respect to all the
       other members.
    4. If medoids have changed, go back to step 2.

    Notes
    -----
    The three methods differs in the way clusters and medoids are computer.
    * 'memory' (all-memory): Uses the dissimilarity matrix. It is very fast,
      since all operations are done on numpy arrays by numpy functions (highly
      optimized).
    * 'hybrid': Computes distances when needed, without needing to store the
      whole dissimilarity matrix. Uses less memory than 'memory' and still
      uses numpy functions, but needs to compute distances anew every time and
      that adds some additional computation time.
    * 'cpu' (all-computations): Computes distances directly and does not
      restort to numpy functions. It may be very slow, but can also work with
      huge datasets.
      Anyway, if this is the case it would be better to use CLARA or CLARANS,
      that deal with the problem in a reasonable time and still give very good
      results.

    References
    ----------
    .. Leonard Kaufman, Peter J. Rousseeuw, "Finding Groups in Data. An
        Introduction to Cluster Analysis"

    """
    # choose the right implementation (all-memory, hybrid, all-computations)
    get_clusters = _get_clusters(metric, method)
    get_medoid = _get_medoid(metric, method)

    n = data.shape[0]
    # step 1
    # arbitrarily choose k data objects as the initial medoids
    medoids = np.random.choice(n, k, replace=False)

    while True:
        changed = False

        # step 2
        # assign every data object to its closest medoid
        clusterid, _ = get_clusters(data, medoids)
        # make sure every medoid stays in its cluster, otherwise there will be
        # problems with dissimilarities matrix with zeros outside the diagonal
        clusterid[medoids] = medoids

        # step 3
        # recompute the medoid for every cluster as the member of the
        # cluster that minimizes the sum of distances with respect to all the
        # other members
        new_medoids = np.copy(medoids)
        for i, medoid in enumerate(medoids):
            cluster = np.where(clusterid == medoid)[0]
            new_medoid = get_medoid(data, cluster)
            if medoid != new_medoid:
                changed = True
                new_medoids[i] = new_medoid

        # step 4
        # if all the medoids have not changed, we reached convergence and hence
        # the algorithm has finished
        if not changed:
            break
        else:
            medoids = np.copy(new_medoids)

    return get_clusters(data, medoids)


def pam_npass(data, k, metric=False, method='memory', npass=1):
    """Partitioning Around Medoids, a realization of k-medoids.

    Parameters
    ----------
    data : (n,) ndarray or (n, n) ndarray
        Data set or dissimilarity matrix.
    k : int
        Number of desired clusters.
    metric : function, optional
        Function to compute pairwise distances.
    method : {'memory', 'hybrid', 'cpu'}
        Implementation to use.
    npass : int, optional
        The number of times the k-medoids clustering algorithm is performed,
        each time with a different (random) initial condition.

    Returns
    -------
    clusterid : (n,) ndarray
        An array containing the number of the cluster to which each object was
        assigned, where the cluster number is defined as the object number of
        the objects representing the cluster centroid.
    error : float
        The within-cluster sum of distances of the clustering solution.
    nfound : int
        The number of times the optimal solution was found.

    """
    # repeat the k-medoids algorithm npass times and select the best
    clusterid = -1
    error = np.inf
    nfound = 0
    for _ in range(npass):
        new_clusterid, new_error = pam(data, k, metric, method)
        if new_error < error:
            # we found a better solution
            error = new_error
            clusterid = new_clusterid
            nfound = 1
        else:
            if np.array_equal(clusterid, new_clusterid):
                # we found another optimal solution
                nfound += 1

    return clusterid, error, nfound


def clara(data, k, metric, samples=5, sampsize=None):
    """Clustering LARge Applications.

    A simple way to extend PAM for larger data sets.

    Parameters
    ----------
    data : (n,) ndarray
        Data set.
    k : int
        Number of desired clusters.
    metric : function
        Function to compute pairwise distances.
    samples : int, optional
        Number of samples to be drawn from the data set. The default, 5, is
        rather small for historical reasons and we recommend to set samples an
        order of magnitude larger.
    sampsize : int, optional
        Number of objects in each sample. sampsize should be higher than the
        number of clusters (k) and at most the total number of objects.

    Returns
    -------
    clusterid : (n,) ndarray
        An array containing the number of the cluster to which each object was
        assigned, where the cluster number is defined as the object number of
        the objects representing the cluster centroid.
    error : float
        The within-cluster sum of distances of the clustering solution.

    Algorithm
    ---------
    1. Draw a sample of ``sampsize`` objects randomly from the entire data set.
    2. Apply PAM clustering to such sample.
    3. Associate every object of the entire data set to the nearest of the k
       medoids just found.
    4. Compute the within-cluster sum of distance (error) of the clustering
       obtained in the previous step. If this error is lower than the current
       minimum, retain the k medoids found in step 2 as the best set so far.
    5. Repeat step 1-4 ``samples`` times.

    Notes
    -----
    This algorithm makes possible to cluster very large data sets. Anyway, the
    sample drawn at each iteration might not be much representative of the
    data. Hence, sampsize and samples should be chosen accordingly.

    References
    ----------
    .. Leonard Kaufman, Peter J. Rousseeuw, "Finding Groups in Data. An
        Introduction to Cluster Analysis"

    """
    # choose which implementation to use, hybrid or cpu
    get_clusters = _get_clusters(metric, method='cpu')
    dissimilarity_matrix = _dissimilarity_matrix(metric)

    n = data.shape[0]
    data = np.array(data)
    if not sampsize:
        # set the default sampsize value
        sampsize = min(40 + 2*k, n)

    error = np.inf
    clusterid = np.empty(n, dtype=np.uint32)
    for _ in range(samples):
        # step 1
        # draw a sample as a random subset of the original dataset
        subset = np.random.choice(n, sampsize, replace=False)

        # step 2
        # compute the dissimilarity matrix of the sample and apply PAM
        diss = dissimilarity_matrix(data[subset])
        partial_clusterid = pam(diss, k)[0]
        medoids = subset[np.unique(partial_clusterid)]

        # step 3 and 4
        # associate each object of the data set to the nearest medoid and
        # compute the error of the clustering
        new_clusterid, new_error = get_clusters(data, medoids)

        # keep the new clustering only if it has a lower error
        if new_error < error:
            error = new_error
            clusterid = new_clusterid

    return clusterid, error


def _clarans(metric):
    """Clustering Large Applications based on RANdomized Search."""
    # choose which implementation to use, hybrid or cpu
    get_clusters = _get_clusters(metric, method='cpu')

    @jit(nopython=True)
    def clarans(data, k, numlocal, maxneighbor):
        """Clustering Large Applications based on RANdomized Search.

        Parameters
        ----------
        data : (n,) ndarray
            Data set.
        k : int
            Number of desired clusters.
        metric : function
            Function to compute pairwise distances.
        numlocal : int
            Number of times to repeat the search for other local minima.
        maxneighbor : int
            Maximum number of the neighbors to look at.

        Returns
        -------
        clusterid : (n,) ndarray
            An array containing the number of the cluster to which each object
            was assigned, where the cluster number is defined as the object
            number of the objects representing the cluster centroid.
        error : float
            The within-cluster sum of distances of the clustering solution.

        Algorithm
        ---------
        1. Choose an arbitrary node from the data set.
        2. Consider a random neighbor of the current node.
        3. If the random neighbor has a lower error than the current node, set
           it as the current node.
        4. Repeat step 2-3 ``maxneighbor`` times.
        5. Repeat step 1-4 ``numlocal`` times and retain the best clustering.

        Notes
        -----
        The best way to explain CLARANS is via a graph abstraction. In fact,
        the process of finding k medoids can be viewed abstractly as searching
        through a certain graph. In this graph, a set of k objects is called
        node. Two nodes are neighbors if their sets differ by only one object.
        Since a node represent a collection of k objects, they can be seen as
        medoids and hence induce a clustering.
        Each node can be assigned an error that is defined to be the total
        dissimilarity (i.e. sum of distances) between every object and the
        medoid of its cluster.

        References
        ----------
        .. R.T. Ng, Jiawei Han, "CLARANS: a method for clustering objects for
        spatial data mining"

        """
        n = data.shape[0]
        choices = np.arange(n)
        best_medoids = np.empty(k, dtype=np.uint32)
        best_error = np.inf
        min_dist = 0

        for _ in range(numlocal):
            # step 1
            # choose an arbitrary node as starting medoids and compute its error
            medoids = np.empty(k, dtype=np.uint32)
            for i in range(k):
                np.random.shuffle(choices)
                medoids[i] = choices[-1]
                choices = choices[:-1]
            error = 0
            for i in range(n):
                min_dist = np.inf
                for med in medoids:
                    dist = metric(data[i], data[med])
                    if dist < min_dist:
                        min_dist = dist
                error += min_dist

            for _ in range(maxneighbor):
                # step 2
                # find a random neighbor, i.e. change only one of the medoids with
                # a random object (that is not already a medoid) of the whole data
                # set
                random_neigh = np.copy(medoids)
                np.random.shuffle(choices)
                non_med = choices[-1]
                non_med_i = np.random.choice(k)
                random_neigh[non_med_i] = non_med

                # step 3
                # compute the error of the random neighbor and compare it with the
                # current node (i.e. current medoids)
                new_error = 0
                for i in range(n):
                    min_dist = np.inf
                    for j, med in enumerate(random_neigh):
                        dist = metric(data[i], data[med])
                        if dist < min_dist:
                            min_dist = dist
                new_error += min_dist
                # choose the induced clustering with lower error
                if new_error < error:
                    error = new_error
                    choices[-1] = medoids[non_med_i]
                    medoids = random_neigh

            # retain the clustering solution with the lowest error
            if error < best_error:
                best_error = error
                best_medoids = medoids

        return get_clusters(data, best_medoids)

    return clarans
