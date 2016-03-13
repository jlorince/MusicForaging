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
import pandas as pd

### CONFIGURATION
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)

random_sample_size=10000000
k_range = np.arange(100,301,10)
#reg_range = [1e-12,1e-8,1e-04,0.01,0.1,0.5,1.0]
reg_range = [1e-12,1e-8]#,1e-8,1e-04,0.01,0.1,0.5,1.0]
n_cores = 24
d = 'MF_tests/'
N=1000

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

######################################################


# vocab_idx = {}
# for line in open('vocab_idx'):
#     line = line.strip().split('\t')
#     vocab_idx[line[0]] = int(line[1])

if False:
    docs = gl.SArray("doc_array")
    docs = gl.SFrame(docs).add_row_number()

    def process(row):
        result = []
        total =float(row['X1'].sum())
        for i,a in enumerate(row['X1']):
            result.append((row['id'],vocab_idx[a],int(row['X1'][a])))
        return result

    result = docs.apply(process)
    result2 = np.vstack([np.array(i) for i in result])

    import numpy as np
    result2 = np.load('/home/jlorince/gl_format.npy')
    with open('/home/jlorince/mf_format.txt','w') as out:
        for i in result2:
            out.write(','.join(map(str,i))+'\n')

    data.rename({'X1':"user_id",'X2':"item_id",'X3':"scrobbles"})
    data.head()
    data.save('gl_sf')

data = gl.SFrame('gl_sf')
data = data[data['scrobbles']>10]
#data['scrobbles'] = data['scrobbles'].apply(lambda x: min(5,np.log10(x)+1))
data['scrobbles'] = data['scrobbles'].apply(lambda x: min(4,np.log10(x)))


artist_indices = {}
id_idx = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    id_idx[int(line[1])] = line[0]
    artist_indices[line[0]] = int(line[1])
lastfm_data = np.zeros((112312,100),dtype=int)
lastfm_data.fill(-1)
with open('lastfm_top_similar_artists') as fin:
    for line in fin:
        try:
            a,top100 = line.strip().split('\t')
        except ValueError:
            continue
        aid = artist_indices.get(a)
        if aid is not None:
            for i,sim in enumerate(top100.split()):
                lastfm_data[aid,i] = artist_indices.get(sim,-1)

def proc_fm(row):
    return len(set(row['current'][:100]).intersection(set(row['fm'])))


combined = gl.SFrame({'fm':lastfm_data})

# if k_range[0]==10:
#     last = None
# else:
#     last = gl.SArray(d+'knn_%s_%s' % (N,k_range[0]-10))
with open(d+'log','a') as fout,open(d+'rmse_log','a') as rmse_log:
    for k in k_range:
        for reg in reg_range:
            ddir = d+'knn_%s_%s_%s' % (N,k,reg)
            if os.path.exists(ddir):
                #topN = gl.SArray(ddir)
                continue
            else:
                model = gl.recommender.factorization_recommender.create(observation_data=data,target='scrobbles',num_factors=k,nmf=True,regularization=reg)
                predicted = model.predict(data)
                rmse = gl.evaluation.rmse(data['scrobbles'],predicted)
                rmse_log.write(str(k)+'\t'+str(rmse)+'\n')
                rmse_log.flush()

                #result = model.get_similar_items(items=range(112312),k=1000)
                #topN = result[['item_id','similar','rank']].unstack(('rank','similar'),new_column_name='knn').sort('item').apply(lambda row: [row['knn'][i]  for i in xrange(1,N+1)])
                result = gl.nearest_neighbors.create(dataset=model.get('coefficients')['item_id'],label='item_id',features=['factors'],distance='cosine').similarity_graph(k=N,output_type='SFrame')
                topN = result[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i]  for i in xrange(1,N+1)])
                topN.save(ddir)



            #combined = gl.SFrame({'current':topN, 'fm':lastfm_data})
            combined['current'] = topN
            overlap = combined.apply(lambda row: proc_fm(row))
            summary = pd.Series(overlap).describe()
            fout.write('\t'.join(map(str,['fm',k,reg]+list(summary)))+'\n')
            fout.flush()

            #if last is not None:
            if False:
                combined = gl.SFrame({'current':topN, 'prev':last})
                overlap = combined.apply(lambda row: len(set(row['current']).intersection(set(row['prev']))))
                summary = pd.Series(overlap).describe()
                fout.write('\t'.join(map(str,['prev','all',k,N]+list(summary)))+'\n')
                fout.flush()
                # for label,cnt in (('sample',random_sample),('top1000',xrange(1000)),('top10000',xrange(10000)),('top50000',xrange(50000))):
                #     summary = pd.Series(overlap).iloc[cnt].describe()
                #     fout.write('\t'.join(map(str,['prev',label,k,N]+list(summary)))+'\n')
                #     fout.flush()

                overlap_100 = combined.apply(lambda row: len(set(row['current'][:100]).intersection(set(row['prev'][:100]))))
                summary = pd.Series(overlap_100).describe()
                fout.write('\t'.join(map(str,['prev','all',k,100]+list(summary)))+'\n')
                fout.flush()
                # for label,cnt in (('sample',random_sample),('top1000',xrange(1000)),('top10000',xrange(10000)),('top50000',xrange(50000))):
                #     summary = pd.Series(overlap_100).iloc[cnt].describe()
                #     fout.write('\t'.join(map(str,['prev',label,k,100]+list(summary)))+'\n')
                #     fout.flush()




