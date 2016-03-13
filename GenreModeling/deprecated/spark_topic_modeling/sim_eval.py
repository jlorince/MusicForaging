import numpy as np
import os
from scipy import sparse
import graphlab as gl
from scipy.spatial.distance import cosine,euclidean
import time
import datetime
import itertools
from operator import itemgetter
import math
import random
import multiprocessing as mp
import datetime

# gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
# gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
# gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
# gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)

def parse_parts(d):
    output = []
    files = os.listdir(d)
    for fi in files:
        output+=[eval(line)[1] for line in open(d+'/'+fi).readlines()]
    return np.array(output)

def parse_sparse_array(arr_string):
    evaluated = eval(arr_string)
    #user_id = evaluated[0]
    data = evaluated[1][2]
    indices = evaluated[1][1]
    length = evaluated[1][0]
    sparse_matrix = sparse.csr_matrix((data,([0]*len(indices),indices)),shape=(1,length))
    return sparse_matrix.todense().A.flatten()

def get_token_counts(arr_string):
    evaluated = eval(arr_string)
    data = evaluated[1][2]
    return sum(data)

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


LDA_vectors_path = "LDA_vectors"

random_sample_size=10000000

np.random.seed(99)
comps = set()
while len(comps)<random_sample_size:
    a = np.random.randint(0,112312)
    b= np.random.randint(0,112312)
    if a!=b:
        comp = tuple(sorted([a,b]))
        comps.add(comp)
comps = tuple(enumerate(comps))

n_cores = 32
chunksize = int(math.ceil(len(comps)/n_cores))
jobs = tuple(slice_iterable(comps, chunksize))

artist_dict = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    artist_dict[line[0]] = int(line[1])


#LDA_vectors = np.array(gl.SArray(LDA_vectors_path).filter(lambda x: x!='').apply(parse_sparse_array))
#token_counts = LDA_vectors.sum(1)[:,np.newaxis]
token_counts = np.array(gl.SArray(LDA_vectors_path).filter(lambda x: x!='').apply(get_token_counts))[:,np.newaxis]


#artist_topic_path = "scala_lda_tf/artist_topic_10"
#user_topic_path = "scala_lda_tf/user_topic_10"
for k in np.arange(50,206,5):
    print '---------%s---------' % k
    print 'calculating matrices'
    artist_topic_path = "scala_lda_tf_50iter/artist_topic_"+str(k)
    user_topic_path = "scala_lda_tf_50iter/user_topic_"+str(k)

    artist_topic = np.loadtxt(artist_topic_path)
    artist_topic /= artist_topic.sum(0)

    user_topic = parse_parts(user_topic_path)

    topic_counts = (user_topic * token_counts).sum(0)
    artist_topic_freq = artist_topic*topic_counts
    artist_topic_probs = artist_topic_freq/artist_topic_freq.sum(1,keepdims=True)


    print 'Running samples'
    def worker(enumerated_comps):
        return [(ind, cosine(artist_topic_probs[a], artist_topic_probs[b]),euclidean(artist_topic_probs[a], artist_topic_probs[b])) for ind, (a, b) in enumerated_comps]
    pool = mp.Pool(processes=n_cores)
    start = time.time()
    work_res = pool.map_async(worker, jobs)
    dists = np.array(map(itemgetter(1,2), sorted(itertools.chain(*work_res.get()))))
    finish = time.time()
    print str(datetime.timedelta(seconds=(finish-start)))


    #print "euclidean (k=%s): mean=%s, var=%s" % (k,e_dists.mean(),e_dists.var())
    #print "cosine (k=%s): mean=%s, var=%s" % (k,c_dists.mean(),c_dists.var())
    np.save('dists_'+str(k), dists)
    pool.terminate()
    #np.save('c_'+str(k), c_dists)


import numpy as np
def rmse(a,b):
    return np.sqrt(((a-b)**2).mean())


rmse_vals = []
current = 10
last_data = None
current_data = np.load('dists_'+str(current)+'.npy')
for k in np.arange(15,206,5):
    last_data = current_data
    try:
        current_data = np.load('dists_'+str(k)+'.npy')
    except:
        break
    result = rmse(current_data[:,0],last_data[:,0]),rmse(current_data[:,1],last_data[:,1])
    rmse_vals.append(result)
    print "RMSE %s <=> %s: %.04f (cosine) %.04f (euclidean)" % (str(k),str(current-5),result[0],result[1])

# print "sanity checks..."
# print "Tool <=> A Perfect Circle : e->%.02f, c->%.02f" % (euclidean(artist_topic_probs[artist_dict['tool']],artist_topic_probs[artist_dict['a+perfect+circle']]), cosine(artist_topic_probs[artist_dict['tool']],artist_topic_probs[artist_dict['a+perfect+circle']]))
# print "Miley Cyrus <=> Demi Lovato : e->%.02f, c->%.02f" % (euclidean(artist_topic_probs[artist_dict['miley+cyrus']],artist_topic_probs[artist_dict['demi+lovato']]), cosine(artist_topic_probs[artist_dict['miley+cyrus']],artist_topic_probs[artist_dict['demi+lovato']]))
# print "The Mars Volta <=> At the Drive-in : e->%.02f, c->%.02f" % (euclidean(artist_topic_probs[artist_dict['the+mars+volta']],artist_topic_probs[artist_dict['at+the+drive-in']]), cosine(artist_topic_probs[artist_dict['the+mars+volta']],artist_topic_probs[artist_dict['at+the+drive-in']]))

# print "Tool <=> Miley Cyrus : e->%.02f, c->%.02f" % (euclidean(artist_topic_probs[artist_dict['tool']],artist_topic_probs[artist_dict['miley+cyrus']]), cosine(artist_topic_probs[artist_dict['tool']],artist_topic_probs[artist_dict['miley+cyrus']]))
# print "The Mars Volta <=> Demi Lovato : e->%.02f, c->%.02f" % (euclidean(artist_topic_probs[artist_dict['the+mars+volta']],artist_topic_probs[artist_dict['demi+lovato']]), cosine(artist_topic_probs[artist_dict['the+mars+volta']],artist_topic_probs[artist_dict['demi+lovato']]))
# print "A Perfect Circle <=> At the Drive-in : e->%.02f, c->%.02f" % (euclidean(artist_topic_probs[artist_dict['at+the+drive-in']],artist_topic_probs[artist_dict['a+perfect+circle']]), cosine(artist_topic_probs[artist_dict['at+the+drive-in']],artist_topic_probs[artist_dict['a+perfect+circle']]))

# e_dists = []
# c_dists = []
# for a,b in comps:
#     e_dists.append(euclidean(artist_topic_probs[a],artist_topic_probs[b]))
#     c_dists.append(cosine(artist_topic_probs[a],artist_topic_probs[b]))
# e_dists = np.array(e_dists)
# c_dists = np.array(c_dists)



