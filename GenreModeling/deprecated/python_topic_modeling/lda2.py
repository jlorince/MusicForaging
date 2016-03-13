import graphlab as gl
import numpy as np
from scipy.spatial.distance import cosine,euclidean
import time
import datetime
from operator import itemgetter
import itertools
import math
import multiprocessing as mp
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)


tfidf = False
n_iter = 50
random_sample_size=5000000
k_range = np.arange(285,301,5)
filename = 'tf'
np.random.seed(99)
n_cores = 16


docs = gl.SArray("doc_array")
if tfidf:
    docs = gl.text_analytics.tf_idf(docs)
    docs.apply(lambda row: {k:round(v)+1 for k,v in row.iteritems()})
train,test = gl.text_analytics.random_split(docs,0.1)
train.save("train_data_"+filename)
test.save("test_data_"+filename)
docs = train

#token_counts = np.array(docs['X1'].apply(lambda row: sum(row.values())))[:,np.newaxis]
token_counts = np.array(docs.apply(lambda row: sum(row.values())))[:,np.newaxis]

comps = set()
while len(comps)<random_sample_size:
    a = np.random.randint(0,112312)
    b= np.random.randint(0,112312)
    if a!=b:
        comp = tuple(sorted([a,b]))
        comps.add(comp)
comps = tuple(enumerate(comps))

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

chunksize = int(math.ceil(len(comps)/n_cores))
jobs = tuple(slice_iterable(comps, chunksize))


with open('log+'+filename,'a') as log:
    for k in k_range:
        overall_start = time.time()
        start = time.time()
        topic_model = gl.topic_model.create(docs,num_topics=k,num_iterations=n_iter,method='cgs')
        topic_model.save("model_"+filename+'_'+str(k))
        print 'model run complete in %s' % str(datetime.timedelta(seconds=(time.time()-start)))
        start = time.time()
        perplexity = topic_model.evaluate(test)['perplexity']
        print 'perplexity calculated in %s' % str(datetime.timedelta(seconds=(time.time()-start)))
        result = '\t'.join(map(str,[k,topic_model.get('alpha'),topic_model.get('beta'),perplexity]))
        print result
        log.write(result+'\n')
        log.flush()
        start = time.time()
        #similarity_sampling(topic_model,k)
        artist_dict = {a:i for i,a in enumerate(topic_model['vocabulary'])}
        doc_topic_mat = np.array(topic_model.predict(docs,output_type='probabilities'))
        word_topic_mat = np.array(topic_model['topics']['topic_probabilities'])
        topic_counts = (doc_topic_mat * token_counts).sum(0)
        word_topic_mat_freq = word_topic_mat*topic_counts
        artist_topic_probs = word_topic_mat_freq/word_topic_mat_freq.sum(1,keepdims=True)

        print 'Running samples'
        def worker(enumerated_comps):
            return [(ind, cosine(artist_topic_probs[a], artist_topic_probs[b]),euclidean(artist_topic_probs[a], artist_topic_probs[b])) for ind, (a, b) in enumerated_comps]
        pool = mp.Pool(processes=n_cores)
        start = time.time()
        work_res = pool.map_async(worker, jobs)
        dists = np.array(map(itemgetter(1,2), sorted(itertools.chain(*work_res.get()))))
        finish = time.time()
        print str(datetime.timedelta(seconds=(finish-start)))
        print "euclidean (k=%s): mean=%s, var=%s" % (k,dists[:,1].mean(),dists[:,1].var())
        print "cosine (k=%s): mean=%s, var=%s" % (k,dists[:,0].mean(),dists[:,1].var())
        np.save('dists_'+str(k), dists)
        pool.terminate()

        start = 10
        last_data = None
        current_data = np.load('dists_'+str(start)+'.npy')
        rmse_vals = []
        for k in np.arange(15,301,5):
            last_data = current_data
            try:
                current_data = np.load('dists_'+str(k)+'.npy')
            except:
                break
            result = rmse(current_data[:,0],last_data[:,0]),rmse(current_data[:,1],last_data[:,1])
            rmse_vals.append(result)
            print "RMSE %s <=> %s: %.04f (cosine) %.04f (euclidean)" % (str(k),str(k-5),result[0],result[1])

        print 'overall time for this test: %s' % str(datetime.timedelta(seconds=(time.time()-overall_start)))


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
data=pd.read_table('../log.txt',header=None,names=["K","alpha","beta","perplexity"],sep=r"\s*")
rmse_vals = [(0.0,0.0)]+rmse_vals
data['cosine'] = [i[0] for i in rmse_vals]
data['euclidean'] = [i[1] for i in rmse_vals]
data = data.set_index('K')[['perplexity','cosine','euclidean']]

