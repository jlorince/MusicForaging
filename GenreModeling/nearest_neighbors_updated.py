import pylast
import pandas as pd
from urllib import quote_plus,unquote_plus
import graphlab as gl
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from numpy.linalg import norm
import multiprocessing as mp
from operator import itemgetter
import math
import itertools
import numpy as np
import time
import os
import time,datetime

gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)


d_lda = 'lda_tests_100iter/'
#d_mf = 'MF_tests/'
d_mf = 'lda_tests_artists/'
k_range = np.arange(140,201,10)
lastfm_data_dir = 'lastfm_top_similar_artists_new'

N = 250
n_cores=32
np.random.seed(101)
thresholds = [25,50,100,150,200,250]
topN_vals = [25,50,100,150,200,250]

thresh_top_combos = [(top,thresh) for top in topN_vals for thresh in thresholds if thresh<=top]

def proc_fm(row,topNs=topN_vals,thresholds=thresholds):
    fm = np.array(row['fm'])
    result = {}
    result[str((N,'all'))] = len(set(row['current']).intersection(fm))/float(N)

    fm_set = set(fm)
    fm_set.discard(-1)
    if len(fm_set)>0:
        result[str((N,'possible'))] = len(set(row['current'][:N]).intersection(fm_set))/float(len(fm_set))

    for t in topNs:
        fm_set = set(fm[:t])
        fm_set.discard(-1)
        if len(fm_set)>0:
            for thresh in thresholds:
                if (thresh<=t) and (len(fm_set)>=thresh):
                    result[str((t,thresh))] = len(set(row['current'][:t]).intersection(fm_set))/float(len(fm_set))
    return result

### Process Last.fm data:
artist_indices = {}
id_idx = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    id_idx[int(line[1])] = line[0]
    artist_indices[line[0]] = int(line[1])
lastfm_data = np.zeros((112312,N),dtype=int)
lastfm_data.fill(-1)
with open(lastfm_data_dir) as fin:
    for line in fin:
        try:
            a,top = line.strip().split('\t')
        except ValueError:
            continue
        aid = artist_indices.get(a)
        if aid is not None:
            for i,sim in enumerate(top.split()):
                lastfm_data[aid,i] = artist_indices.get(sim,-1)


out_file_lda = d_lda+'log_knn'
out_file_mf = d_mf+'log_knn'
combined = gl.SFrame({'fm':lastfm_data})

for out_file in (out_file_mf,out_file_lda):
    if not os.path.exists(out_file):
        with open(out_file,'w') as fout:
            fout.write('\t'.join(map(str,['source', 'k', 'method', 'topN','count', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max']))+'\n')


model_dict = {'mf':out_file_mf}#,'lda':out_file_lda}

for m in model_dict:

    if m == 'mf':
        d = d_mf
    elif m == 'lda':
        d = d_lda

    if k_range[0]==10:
        last = None
    else:
        result = gl.SFrame(d+'knn_{}'.format(k_range[0]-10))
        last = result[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i] for i in xrange(1,N+1)])

    for k in k_range:


        with open(model_dict[m],'a') as fout:
            print 'K=%s' % k

            # comparison to previous model
            ddir = d+'knn_{}'.format(k)
            if os.path.exists(ddir):
                result = gl.SFrame(ddir)
            else:
                features = gl.SFrame(np.load(d+'features_'+str(k)+'.npy'))
                result = gl.nearest_neighbors.create(dataset=features,distance='cosine').similarity_graph(k=N,output_type='SFrame')
                result.save(ddir)
            topN = result[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i] for i in xrange(1,N+1)])

            combined['current'] = topN
            overlap = combined.apply(proc_fm)
            for top_measure in [(N,'all'),(N,'possible')]+thresh_top_combos:
                top,measure = top_measure
                summary = pd.Series([row.get(str(top_measure)) for row in overlap]).dropna().describe()
                fout.write('\t'.join(map(str,['fm',k,measure,top]+list(summary)))+'\n')
                fout.flush()


            if last is not None:
                combined_prev = gl.SFrame({'current':topN,'prev':last})
                for t in topN_vals:
                    overlap = combined_prev.apply(lambda row: len(set(row['current'][:t]).intersection(set(row['prev'][:t])))/float(t))
                    summary = pd.Series(overlap).dropna().describe()
                    fout.write('\t'.join(map(str,['prev',k,None,t]+list(summary)))+'\n')
                    fout.flush()

            # split comparison
            if m=='lda':
                ddir_a = d+'knn_{}_randa'.format(k)
                ddir_b = d+'knn_{}_randb'.format(k)
                if os.path.exists(ddir_a) and os.path.exists(ddir_b):
                    result_a = gl.SFrame(ddir_a)
                    result_b = gl.SFrame(ddir_b)
                else:
                    features_a = gl.SFrame(np.load(d+'features_rand_a_'+str(k)+'.npy'))
                    features_b = gl.SFrame(np.load(d+'features_rand_b_'+str(k)+'.npy'))
                    result_a = gl.nearest_neighbors.create(dataset=features_a,distance='cosine').similarity_graph(k=N,output_type='SFrame')
                    result_a.save(ddir_a)
                    result_b = gl.nearest_neighbors.create(dataset=features_b,distance='cosine').similarity_graph(k=N,output_type='SFrame')
                    result_b.save(ddir_b)
                topN_a = result_a[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i] for i in xrange(1,N+1)])
                topN_b = result_b[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i] for i in xrange(1,N+1)])


                split_combined = gl.SFrame({'a':topN_a, 'b':topN_b})

                for t in topN_vals:
                    overlap = split_combined.apply(lambda row: len(set(row['a'][:t]).intersection(set(row['b'][:t])))/float(t))
                    summary = pd.Series(overlap).dropna().describe()
                    fout.write('\t'.join(map(str,['split',k,None,t]+list(summary)))+'\n')
                    fout.flush()

            last = topN


