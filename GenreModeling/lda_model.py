import graphlab as gl
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
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)
n_iter = 50
random_sample_size=10000000
k_range = np.arange(200,201,10)
n_cores = 24


d = 'lda_tests_artists/'
doc_basis = 'artist'

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

### JENSEN SHANNON DIVERGENCE - WE USED THIS BEFORE SO MUST TAKE SQRT BELOW!!!
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
    #return np.sqrt(0.5 * (entropy(_P, _M) + entropy(_Q, _M)))

######################################################

# if we've already built doc_array, load it
if doc_basis == 'user':
    if os.path.exists("doc_array"):
        docs = gl.SArray("doc_array")
# otherwise generate the doc_array from the raw LDA vectors from Spark
    else:
        raise("This should no longer be handled by this script")
        print "loading data..."
        raw_docs = gl.SArray("LDA_vectors/")
        vocab_idx = {}
        for line in open('vocab_idx'):
            line = line.strip().split('\t')
            vocab_idx[int(line[1])] = line[0]
            #vocab_idx[line[0]] = int(line[1])

        def formatter(row):
            row = eval(row)
            result = dict(zip([vocab_idx[term] for term in row[1][1]],row[1][2]))
            return (row[0],result)

        print "formatting data..."
        docs = raw_docs.filter(lambda x: x!="").apply(formatter)
        print "filtering data..."
        docs = docs.filter(lambda x: sum(x[1].values())>=1000)
        user_idx = {row[0]:i for i,row in enumerate(docs)}
        with open('user_idx','w') as fout:
            for k,v in user_idx.iteritems():
                fout.write(str(k)+'\t'+str(v)+'\n')
        print "saving data..."
        docs.save("doc_array",format='binary')

elif doc_basis == 'artist':
    if os.path.exists("doc_sframe_artists"):
        docs = gl.SFrame("doc_sframe_artists")['users']
        #rand_a = gl.SArray(d+"rand_a")
        #rand_b = gl.SArray(d+"rand_b")
    else:
        data = gl.SFrame.read_csv('mf_format.txt',header=False)
        unique_users = list(data['X1'].unique())
        shuffle(unique_users)
        l = len(unique_users)
        a = unique_users[:l/2]
        #b = unique_users[l/2:]

        rand_a = data.filter_by(a,'X1',exclude=False).groupby('X2',{"users":gl.aggregate.CONCAT("X1","X3")})['users']
        rand_b = data.filter_by(a,'X1',exclude=True).groupby('X2',{"users":gl.aggregate.CONCAT("X1","X3")})['users']
        rand_a = rand_a.apply(lambda d: {str(k):d[k] for k in d})
        rand_b = rand_b.apply(lambda d: {str(k):d[k] for k in d})
        rand_a.save(d+'rand_a')
        rand_b.save(d+'rand_b')

        grp = data.groupby('X2',{"users":gl.aggregate.CONCAT("X1","X3")}).sort('X2')
        grp.save("doc_sframe_artists",format='binary')

        docs = grp['users']


# Generate train/test data for main LDA model
print "processing train/test split..."
if os.path.exists(d+"train_data") and os.path.exists(d+"test_data"):
    train = gl.SArray(d+"train_data")
    test = gl.SArray(d+"test_data")
else:
    train,test = gl.text_analytics.random_split(docs,0.1)
    if doc_basis == 'artist':
        train = train.apply(lambda d: {str(k):d[k] for k in d})
        test = test.apply(lambda d: {str(k):d[k] for k in d})
    train.save(d+"train_data")
    test.save(d+"test_data")

# generate random 50/50 split of data for secondary test
print "generating 50/50 split..."
if doc_basis == 'user':
    if os.path.exists(d+"rand_a") and os.path.exists(d+"rand_b"):
        rand_a = gl.SArray(d+"rand_a")
        rand_b = gl.SArray(d+"rand_b")
    else:
        rand_a, rand_b = gl.SFrame(docs).random_split(0.5)
        rand_a = rand_a['X1']
        rand_b = rand_b['X1']
        rand_a.save(d+'rand_a')
        rand_b.save(d+'rand_b')


if doc_basis == 'user':
    # precalculate token counts:
    print "generating token counts..."
    if os.path.exists(d+"token_counts_train.npy"):
        token_counts_train = np.load(d+'token_counts_train.npy')
    else:
        token_counts_train = np.array(train.apply(lambda row: sum(row.values())))[:,np.newaxis]
        np.save(d+'token_counts_train.npy',token_counts_train)
    if os.path.exists(d+"token_counts_rand_a.npy"):
        token_counts_rand_a = np.load(d+'token_counts_rand_a.npy')
    else:
        token_counts_rand_a = np.array(rand_a.apply(lambda row: sum(row.values())))[:,np.newaxis]
        np.save(d+'token_counts_rand_a.npy',token_counts_rand_a)
    if os.path.exists(d+"token_counts_rand_b.npy"):
        token_counts_rand_b = np.load(d+'token_counts_rand_b.npy')
    else:
        token_counts_rand_b = np.array(rand_b.apply(lambda row: sum(row.values())))[:,np.newaxis]
        np.save(d+'token_counts_rand_b.npy',token_counts_rand_b)

# generate comps for testing
print "generating comps..."
if os.path.exists(d+"random_comps"):
    comps = cPickle.load(open(d+'random_comps'))
else:
    comps = set()
    while len(comps)<random_sample_size:
        a = np.random.randint(0,112312)
        b = np.random.randint(0,112312)
        if a!=b:
            comp = tuple(sorted([a,b]))
            comps.add(comp)
    cPickle.dump(comps,open(d+'random_comps','w'))

comps = tuple(enumerate(comps))
chunksize = int(math.ceil(len(comps)/n_cores))
jobs = tuple(slice_iterable(comps, chunksize))

### this is a little hinky, but needed for the parallel implementation of the distance calculations

def dist_prep(model,predict_data,token_counts=None):
    if doc_basis == 'user':
        artist_dict = {a:i for i,a in enumerate(model['vocabulary'])}
        seq = np.array([artist_dict[artist_indices[i]] for i in sorted(artist_indices)])
        doc_topic_mat = np.array(model.predict(predict_data,output_type='probabilities')) # returns a user x topic matrix
        word_topic_mat = np.array(model['topics']['topic_probabilities'])[seq] # returns a word x topic matrix
        topic_counts = (doc_topic_mat * token_counts).sum(0)
        word_topic_mat_freq = word_topic_mat*topic_counts
        artist_topic_probs = word_topic_mat_freq/word_topic_mat_freq.sum(1,keepdims=True)
    elif doc_basis == 'artist':
        artist_topic_probs = np.array(model.predict(predict_data,output_type='probabilities'))
    def worker(enumerated_comps):
        return [(ind, cosine(artist_topic_probs[a], artist_topic_probs[b]),euclidean(artist_topic_probs[a], artist_topic_probs[b]),JSD(artist_topic_probs[a], artist_topic_probs[b])) for ind, (a, b) in enumerated_comps]
    return worker,artist_topic_probs


def calc_dists():
    print 'Running samples'
    pool = mp.Pool(processes=n_cores)
    start = time.time()
    work_res = pool.map_async(worker, jobs)
    dists = np.array(map(itemgetter(1,2,3), sorted(itertools.chain(*work_res.get()))))
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


if not os.path.exists(d+'log'):
    with open(d+'log','w') as log:
        log.write('\t'.join(["K", "perplexity", "rank_corr_prev_cosine", "rank_corr_prev_euclidean","rank_corr_prev_jsd","rmse_prev_cosine", "rmse_prev_euclidean","rmse_prev_jsd","split_rank_corr_cosine", "split_rank_corr_euclidean", "split_rank_corr_jsd","split_rmse_cosine", "split_rmse_euclidean","split_rmse_jsd","jsd_cosine_corr","jsd_euclidean_corr"])+'\n')


with open(d+'log','a') as log:
    for k in k_range:

        # standard topic model
        overall_start = time.time()
        model_name = "model_"+str(k)
        if os.path.exists(d+model_name):
            topic_model = gl.load_model(d+model_name)
        else:
            topic_model = gl.topic_model.create(train,num_topics=k,num_iterations=n_iter,method='cgs')
            topic_model.save(d+model_name)
            print 'model training time: %s' % topic_model.get('training_time')
        perplexity = topic_model.evaluate(test)['perplexity']

        # distances for standard topic model
        if os.path.exists(d+'dists_'+str(k)+'.npy'):
            dists = np.load(d+'dists_'+str(k)+'.npy')
        else:
            if doc_basis == 'user':
                worker,artist_topic_probs = dist_prep(topic_model,train,token_counts_train)
            elif doc_basis == 'artist':
                worker,artist_topic_probs = dist_prep(topic_model,train)

            np.save(d+'features_'+str(k),artist_topic_probs)
            dists = calc_dists()
            np.save(d+'dists_'+str(k)+'.npy', dists)
        if last is not None:
            prev_comp_cosine = spearmanr(last[:,0],dists[:,0]).correlation
            prev_comp_euclidean = spearmanr(last[:,1],dists[:,1]).correlation
            prev_comp_jsd = spearmanr(np.sqrt(last[:,2]),np.sqrt(dists[:,2])).correlation
            prev_rmse_cosine = rmse(last[:,0],dists[:,0])
            prev_rmse_euclidean = rmse(last[:,1],dists[:,1])
            prev_rmse_jsd = rmse(np.sqrt(last[:,2]),np.sqrt(dists[:,2]))


        else:
            prev_comp_cosine = 0.0
            prev_comp_euclidean = 0.0
            prev_comp_jsd = 0.0
            prev_rmse_cosine = 0.0
            prev_rmse_euclidean = 0.0
            prev_rmse_jsd = 0.0
        last = dists

        # split topic model comparison
        # if os.path.exists(d+"model_rand_a_"+str(k)) and os.path.exists(d+"model_rand_b_"+str(k)):
        #     model_a = gl.load_model(d+"model_rand_a_"+str(k))
        #     model_b = gl.load_model(d+"model_rand_b_"+str(k))
        # else:
        #     model_a = gl.topic_model.create(rand_a,num_topics=k,num_iterations=n_iter,method='cgs')
        #     model_a.save(d+"model_rand_a_"+str(k))
        #     print 'model training time: %s' % model_a.get('training_time')
        #     model_b = gl.topic_model.create(rand_b,num_topics=k,num_iterations=n_iter,method='cgs')
        #     model_b.save(d+"model_rand_b_"+str(k))
        #     print 'model training time: %s' % model_b.get('training_time')

        # if os.path.exists(d+'dists_rand_a_'+str(k)+'.npy') and os.path.exists(d+'dists_rand_b_'+str(k)+'.npy'):
        #     dists_a = np.load(d+'dists_rand_a_'+str(k)+'.npy')
        #     dists_b = np.load(d+'dists_rand_b_'+str(k)+'.npy')
        # else:
        #     if doc_basis == 'user':
        #         worker,artist_topic_probs = dist_prep(model_a,rand_a,token_counts_rand_a)
        #     elif doc_basis == 'artist':
        #         worker,artist_topic_probs = dist_prep(model_a,rand_a)
        #     np.save(d+'features_rand_a_'+str(k),artist_topic_probs)
        #     dists_a = calc_dists()
        #     np.save(d+'dists_rand_a_'+str(k)+'.npy',dists_a)
        #     if doc_basis == 'user':
        #         worker,artist_topic_probs = dist_prep(model_b,rand_b,token_counts_rand_b)
        #     elif doc_basis == 'artist':
        #         worker,artist_topic_probs = dist_prep(model_b,rand_b)
        #     np.save(d+'features_rand_b_'+str(k),artist_topic_probs)
        #     dists_b = calc_dists()
        #     np.save(d+'dists_rand_b_'+str(k)+'.npy',dists_b)

        # split_comp_cosine = spearmanr(dists_a[:,0],dists_b[:,0]).correlation
        # split_comp_euclidean = spearmanr(dists_a[:,1],dists_b[:,1]).correlation
        # split_comp_jsd = spearmanr(np.sqrt(dists_a[:,2]),np.sqrt(dists_b[:,2])).correlation
        # split_rmse_cosine = rmse(dists_a[:,0],dists_b[:,0])
        # split_rmse_euclidean = rmse(dists_a[:,1],dists_b[:,1])
        # split_rmse_jsd = rmse(np.sqrt(dists_a[:,2]),np.sqrt(dists_b[:,2]))

        split_comp_cosine = 0 #spearmanr(dists_a[:,0],dists_b[:,0]).correlation
        split_comp_euclidean = 0 #spearmanr(dists_a[:,1],dists_b[:,1]).correlation
        split_comp_jsd = 0 #spearmanr(np.sqrt(dists_a[:,2]),np.sqrt(dists_b[:,2])).correlation
        split_rmse_cosine = 0 #rmse(dists_a[:,0],dists_b[:,0])
        split_rmse_euclidean = 0 #rmse(dists_a[:,1],dists_b[:,1])
        split_rmse_jsd = 0 #rmse(np.sqrt(dists_a[:,2]),np.sqrt(dists_b[:,2]))

        jsd_cosine_corr = spearmanr(np.sqrt(dists[:,2]),dists[:,0]).correlation
        jsd_euclidean_corr = spearmanr(np.sqrt(dists[:,2]),dists[:,1]).correlation

        print 'overall time for this test: %s' % str(datetime.timedelta(seconds=(time.time()-overall_start)))

        result = '\t'.join(map(str,[k,perplexity,prev_comp_cosine,prev_comp_euclidean,prev_comp_jsd,prev_rmse_cosine,prev_rmse_euclidean,prev_rmse_jsd,split_comp_cosine,split_comp_euclidean,split_comp_jsd,split_rmse_cosine,split_rmse_euclidean,split_rmse_jsd,jsd_cosine_corr,jsd_euclidean_corr]))
        print result
        log.write(result+'\n')
        log.flush()

