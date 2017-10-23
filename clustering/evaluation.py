"""Methods to evaluate a clustering solution."""
import numpy as np
from numba import jit
from clustering.metrics import _dissimilarity_matrix_partial


def silhouette_samples_cpu(data, labels, metric):
    """Compute the silhouette for each data object.

    Parameters
    ----------
    data : (n,) ndarray
        Data set.
    labels : (n,) ndarray
        Array of labels that associate each data object to its cluster.
    metric : function
        Function to compute pairwise distances.

    Returns
    -------
    (n,) ndarray
        Silhouette for each data object.

    Notes
    -----
    Construction of silhouettes according to [1].
    This implementation is based only on computations and uses little if no
    memory, but it is quite slow.
    Version to let ``numba`` run in 'nopython' mode (faster).

    References
    ----------
    .. [1] Rousseeuw, Peter. (1987). "Silhouettes: A Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Comput. Appl. Math.
       20, 53-65.

    """
    n = data.shape[0]
    # np.unique is not supported in numba 0.35, so it must go here
    unique_labels = np.unique(labels)

    @jit(nopython=True)
    def _ss(data, labels):
        silhouette = np.empty(n)

        for i in range(n):
            if np.where(labels == labels[i])[0].shape[0] > 1:
                # if the cluster the current object belongs to has more than
                # one element
                a_xi = 0
                b_xi = np.inf
                for curr_label in unique_labels:
                    cluster_members = np.where(labels == curr_label)[0]
                    if curr_label == labels[i]:
                        # compute the coefficient a(i), that is the average
                        # dissimilarity of i to all other objects in its
                        # cluster
                        dist = 0
                        for j in cluster_members:
                            dist += metric(data[i], data[j])
                        # note that we cannot compute simply the mean, because
                        # we do not want to include i into the average
                        # dissimilarity
                        if cluster_members.shape[0] > 1:
                            a_xi = dist / (cluster_members.shape[0] - 1)
                    else:
                        # compute the coefficient b(i), that is the average
                        # dissimilarity of i to all objects of the closest
                        # cluster that does not contain i
                        dist = 0
                        for j in cluster_members:
                            dist += metric(data[i], data[j])
                        # here we compute the mean with no issues
                        b_xi = min(b_xi, dist / cluster_members.shape[0])
                # formula according to [1]
                silhouette[i] = (b_xi - a_xi) / max(a_xi, b_xi)
            else:
                # otherwise, if the current object is the only member of its
                # cluster, according to the paper we set the silhouette to be
                # zero
                silhouette[i] = 0

        return silhouette

    return _ss(data, labels)


@jit
def silhouette_samples(diss, labels):
    r"""Compute the silhouette for each data object.

    Parameters
    ----------
    diss : (n, n) ndarray
        Dissimilarity matrix.
    labels : (n,) ndarray
        Array of labels that associate each data object to its cluster.

    Returns
    -------
    (n,) ndarray
        Silhouette for each data object.

    Notes
    -----
    Construction of silhouettes according to [1].
    This implementation is based on the dissimilarity matrix, that requires to
    store a n\*n matrix in memory but is fast. Roughly the same implementation
    as ``sklearn.metrics.silhouette_samples``, but should run faster thanks to
    ``numba``.

    References
    ----------
    .. [1] Rousseeuw, Peter. (1987). "Silhouettes: A Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Comput. Appl. Math.
       20, 53-65.

    """
    unique_labels = np.unique(labels)
    n = diss.shape[0]
    k = unique_labels.shape[0]

    # for each object i, a(i) is the average dissimilarity of i to all other
    # objects in its cluster (say A)
    a = np.zeros(n)
    # for each object i, b(i) is the average dissimilarity of i to all objects
    # of the closest cluster that does not contain i (say C =/= A)
    b = np.inf + a
    for curr in range(k):
        # extract from the dissimilarity matrix the submatrix with as rows
        # only the objects belonging to the current cluster (selected objects)
        mask = labels == unique_labels[curr]
        sub_diss = diss[mask]

        # compute a(i): we cannot use the mean directly, because we want the
        # average dissimilarity leaving out the selected object; hence, divide
        # the sum of dissimilarities by the number of members of the current
        # cluster minus one
        n_members = mask.sum() - 1
        if n_members != 0:
            # if the current cluster has more than one element
            a[mask] = sub_diss[:, mask].sum(axis=1) / n_members

            for other in range(k):
                if curr != other:
                    # for each other cluster (that will not contain any
                    # currently selected objects) compute the average
                    # dissimilarity
                    other_mask = labels == unique_labels[other]
                    other_distances = sub_diss[:, other_mask].mean(axis=1)
                    # b(i) is the minimum of such average dissimilarities
                    b[mask] = np.minimum(b[mask], other_distances)
        else:
            # if the current cluster has just one element, a is not well
            # defined, hence the corresponding silhouette will be 0
            # note: a = 1, b = 1 => s = (1 - 1) / max(1, 1) = 0
            a[mask] = 1
            b[mask] = 1

    silhouettes = (b - a) / np.maximum(a, b)
    return silhouettes


@jit
def silhouette_samples_partial(diss, labels, start=0):
    r"""Compute the silhouette of a subset of the data.

    Parameters
    ----------
    diss : (m, n) ndarray
        Partial dissimilarity matrix, with ``m`` <= ``n``.
    labels : (n,) ndarray
        Array of labels that associate each data object to its cluster.
    start : int, optional
        Starting index of the given subset of the data with respect to the
        original data indices.

    Returns
    -------
    (m,) ndarray
        Silhouette for each object of the subset.

    Notes
    -----
    This implementation is similar to ``silhouette_samples``, but accepts as
    input also subsets of the original data. Requires to store a m\*n matrix in
    memory instead of a n\*n one. Used with ``silhouettes_samples_batch``.

    References
    ----------
    .. [1] Rousseeuw, Peter. (1987). "Silhouettes: A Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Comput. Appl. Math.
       20, 53-65.

    """
    unique_labels = np.unique(labels)
    m, _ = diss.shape
    k = unique_labels.shape[0]


    a = np.zeros(m)
    b = np.inf + a
    for curr in range(k):
        mask = labels == unique_labels[curr]
        n_members = mask.sum() - 1

        # extract from the partial dissimilarity matrix the submatrix with as
        # rows only the objects belonging to the current cluster and in the
        # current subset of the data
        mask_m = mask[start:start+m]
        sub_diss = diss[mask_m]
        # compute a(i), but only for objects in the current subset of the data
        if n_members != 0:
            a[mask_m] = sub_diss[:, mask].sum(axis=1) / n_members

        for other in range(k):
            if curr != other:
                # compute b(i), but only for objects in the current subset of
                # the data
                other_mask = labels == unique_labels[other]
                other_distances = sub_diss[:, other_mask].mean(axis=1)
                b[mask_m] = np.minimum(b[mask_m], other_distances)

    return (b - a)/np.maximum(a, b)


@jit
def silhouette_samples_batch(data, labels, metric, batch_num):
    r"""Compute the silhouette for each data object in batches.

    Parameters
    ----------
    data : (n,) ndarray
        Data set.
    labels : (n,) ndarray
        Array of labels that associate each data object to its cluster.
    metric : function
        Function to compute pairwise distances.

    Returns
    -------
    (n,) ndarray
        Silhouette for each data object.

    Notes
    -----
    Construction of silhouettes according to [1].
    This implementation is based on dividing the data in batches horizontally,
    so that every batch can be passed to the function
    ``silhouette_samples_partial`` and computed separately. This way, there is
    no need to store the whole n\*n dissimilarity matrix in memory but you
    still get a decent speed.

    References
    ----------
    .. [1] Rousseeuw, Peter. (1987). "Silhouettes: A Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Comput. Appl. Math.
       20, 53-65.

    """
    dissimilarity_matrix_partial = _dissimilarity_matrix_partial(metric)

    n = data.shape[0]
    m = int(n/batch_num)

    silhouettes = np.empty(n)
    start = 0
    for _ in range(batch_num):
        # compute the partial dissimilarity matrix for the current batch
        diss = dissimilarity_matrix_partial(data, start, m)
        # compute the silhouettes for the objects in the current batch
        silhouettes[start:start+m] = silhouette_samples_partial(
            diss, labels, start)
        # move forward to the next batch
        start += m
    if start < n:
        # if m does not divide n, compute the remaining silhouettes
        diss = dissimilarity_matrix_partial(data, start, n-start)
        silhouettes[start:] = silhouette_samples_partial(
            diss, labels, start)

    return silhouettes


def silhouette_plot(silhouettes, labels):
    """Show the silhouette plot."""
    import matplotlib.pyplot as plt

    # remember that labels may not contain values between 0 and k (number of
    # clusters) but k arbitrary distinct values (as it happens for the output
    # of kmedoids)
    unique_labels = np.unique(labels)

    # where to start to plot the silhouettes from the bottom
    y_lower = 10
    for curr_label in unique_labels:
        # get silhouettes for the current cluster and sort them
        curr_silhouettes = silhouettes[labels == curr_label]
        curr_silhouettes.sort()
        n_members = curr_silhouettes.shape[0]

        # fill the appropriate range with the sorted silhouettes
        y_upper = y_lower + n_members
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, curr_silhouettes,
                          alpha=0.7)
        y_lower = y_upper + 10

    # also add a vertical line to indicate the silhouette score
    plt.axvline(silhouettes.mean(), color='r', linestyle='dashed')

    plt.title("Silhouette plot")
    plt.show()
