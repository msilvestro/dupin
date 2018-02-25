"""Process data transforming the original one."""
# pylint: disable=C0103
from math import gcd
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

TYPE = 'mode'
# TYPE = 'lcm'

# lcm calculation
def lcm(a, b):
    """Least common multiple."""
    return a * b // gcd(a, b)

# read the original data
csv_file = 'data/tension_curves.txt'
data = np.asarray([np.asarray(line.split(','), dtype=np.float)
                     for line in open(csv_file)])
n = data.shape[0]

# apply exponential smoothing
data_exp = []
for i, curve in enumerate(data):
    l = curve.shape[0]
    weights = np.array([2**j for j in range(l)])
    numer = np.cumsum(np.multiply(curve, weights))
    denom = np.cumsum(weights)
    data_exp.append(np.divide(numer, denom))
data_exp = np.asarray(data_exp)

# warp the stories according to the mode
def warp(vector, length, kind='linear'):
    n = vector.shape[0]
    linspace = np.linspace(0, 1, n)
    interp = interp1d(linspace, vector, kind=kind)
    new_linspace = np.linspace(0, 1, length)
    return interp(new_linspace)

lens = np.array([curve.shape[0] for curve in data_exp])
if TYPE == 'mode':
    warp_len = stats.mode(lens)[0][0]
elif TYPE == 'lcm':
    unique_lens = np.unique(lens)
    warp_len = 1
    for l in unique_lens:
        warp_len = lcm(warp_len, int(l))
print("Mode: {:}".format(warp_len))

data_warped = np.empty((n, warp_len))
for i, curve in enumerate(data_exp):
    data_warped[i] = warp(curve, warp_len)

if TYPE == 'mode':
    np.savetxt('data/warped_curves.gz', data_warped)
elif TYPE == 'lcm':
    np.savetxt('data/warped_curves_lcm.gz', data_warped)
