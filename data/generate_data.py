"""Generate data transforming the original one."""
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
csv_file = 'data/curves_exp.csv'
data = np.array([np.asarray(line.split(','), dtype=np.float)
                 for line in open(csv_file)])
n = data.shape[0]

# warp the stories in relationship to the mode
def warp(vector, length, kind='linear'):
    n = vector.shape[0]
    linspace = np.linspace(0, 1, n)
    interp = interp1d(linspace, vector, kind=kind)
    new_linspace = np.linspace(0, 1, length)
    return interp(new_linspace)

lens = np.array([story.shape[0] for story in data])
if TYPE == 'mode':
    warp_len = stats.mode(lens)[0][0]
elif TYPE == 'lcm':
    unique_lens = np.unique(lens)
    warp_len = 1
    for l in unique_lens:
        warp_len = lcm(warp_len, int(l))
print(warp_len)

output = np.empty((n, warp_len))
for i, story in enumerate(data):
    output[i] = warp(story, warp_len)

if TYPE == 'mode':
    np.savetxt('data/curves_exp_warped.gz', output)
elif TYPE == 'lcm':
    np.savetxt('data/curves_exp_warped_lcm.gz', output)
