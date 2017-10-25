"""Singular value decomposition for DoppioGioco."""
# pylint: disable=C0103
from time import time
import numpy as np

data = np.loadtxt('data.gz')
n, m = data.shape

start = time()
u, s, v = np.linalg.svd(data)
print("Elapsed time: {:.4f}".format(time() - start))

np.savetxt('results/svd_s.gz', s)
np.savetxt('results/svd_v.gz', v)

S = np.zeros((n, m))
S[:m, :m] = np.diag(s)
w = np.dot(u, S)
np.savetxt('results/svd_w.gz', w)
