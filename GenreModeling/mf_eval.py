import numpy as np
from scipy.spatial.distance import cosine,euclidean
from scipy.stats import spearmanr
import time
import datetime
from operator import itemgetter
import itertools
import math
import multiprocessing as mp
import os
import cPickle
from random import shuffle

### CONFIGURATION
random_sample_size=10000000
k_range = np.arange(10,201,10)
n_cores = 24

split = False


d = 'MF_tests/'

np.random.seed(101)

artist_indices = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    artist_indices[int(line[1])] = line[0]
    #artist_indices[line[0]] = int(line[1])


######################################################
# helper function for parallel similarity calculations

def slice_iterable(iterable, chunk):
    """
    Slices an iterable into chunks of size n
    :param chunk: the number of items per slice
    :type chunk: int
    :type iterable: collections.Iterable
    :rtype: collections.Generator
    """
    _it = iter(iterable)
    return itertools.takewhile(
        bool, (tuple(itertools.islice(_it, chunk)) for _ in itertools.count(0))
    )


print "loading comps..."
comps = cPickle.load(open(d+'random_comps'))
comps = tuple(enumerate(comps))
chunksize = int(math.ceil(len(comps)/n_cores))
jobs = tuple(slice_iterable(comps, chunksize))


def calc_dists():
    print 'Running samples'
    pool = mp.Pool(processes=n_cores)
    start = time.time()
    work_res = pool.map_async(worker, jobs)
    dists = np.array(map(itemgetter(1), sorted(itertools.chain(*work_res.get()))))
    print "distances calculated in %s" % str(datetime.timedelta(seconds=(time.time()-start)))
    pool.terminate()
    return dists

def rmse(a,b):
    return np.sqrt((((a-b)**2).sum())/float(len(a)))

# main testing loop
print "starting main loop..."
if k_range[0]==10:
    last = None
else:
    last = np.load(d+'dists_'+str(k_range[0]-10)+'.npy')



with open(d+'log_eval','a') as log:
    for k in k_range:

        if os.path.exists(d+'dists_'+str(k)+'.npy'):
            dists = np.load(d+'dists_'+str(k)+'.npy')
        else:
            features = np.load(d+'features_{}.npy'.format(k))

            def worker(enumerated_comps):
                return [(ind, cosine(features[a], features[b])) for ind, (a, b) in enumerated_comps]

            dists = calc_dists()
            np.save(d+'dists_'+str(k),dists)

        if last is not None:
            prev_comp_cosine = spearmanr(last,dists).correlation
            prev_rmse_cosine = rmse(last,dists)

        else:
            prev_comp_cosine = 0.0
            prev_rmse_cosine = 0.0
        last = dists

        if split:
            features_a = np.load(d+'features_rand_a{}.npy'.format(k))
            features_b = np.load(d+'features_rand_b{}.npy'.format(k))

            split_comp_cosine = spearmanr(dists_a,dists_b).correlation
            split_rmse_cosine = rmse(dists_a,dists_b)


        result = '\t'.join(map(str,[k,prev_comp_cosine,prev_rmse_cosine]))

        log.write(result+'\n')
        log.flush()

