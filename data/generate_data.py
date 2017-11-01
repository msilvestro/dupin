"""Generate data transforming the original one."""
# pylint: disable=C0103
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

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
mode = stats.mode(lens)[0][0]

output = np.empty((n, mode))
for i, story in enumerate(data):
    output[i] = warp(story, mode)

np.savetxt('data/curves_exp_warped.gz', output)
