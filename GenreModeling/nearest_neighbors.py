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

<<<<<<< HEAD

=======
API_KEY="a7783eed0e7a281f855704fab477a1d3"
API_SECRET="474d26d872a5ac15ab96f7a0aca1bdbb"
outfile = 'data/lastfm_top_similar_artists'

scrape_data = False
calc_nearest_neighbors = True
>>>>>>> 8cd66b95ea5dad24826289a07346f2ba280dcbc4
d = 'lda_tests_100iter/'
k_range = np.arange(150,301,5)

N = 1000
n_cores=32
np.random.seed(101)

sample_size = 1000
random_sample = np.random.choice(xrange(112312),sample_size,replace=False)


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return np.sqrt(0.5 * (entropy(_P, _M) + entropy(_Q, _M)))


def triple_apply_knn(features):

    def graph_JSD(src,edge,dst):
        P = src['X1']
        Q = dst['X1']
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        edge['distance'] = 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
        return (src, edge, dst)

    n = len(features)
    sf = gl.SFrame(features)
    sf = sf.add_row_number('row_id')

    sg = gl.SGraph().add_vertices(sf, vid_field='row_id')

    edges = [gl.Edge(u, v, attr={'distance': None}) for (u, v) in itertools.combinations(range(n), 2)]
    sg = sg.add_edges(edges)

    sg_dist = sg.triple_apply(graph_JSD, mutated_fields=['distance'])

    #knn = sg_dist.edges.groupby("__src_id", {"knn" : gl.aggregate.CONCAT("__dst_id","distance")}).sort("__src_id")
    #top_neighbors = knn.apply(lambda row: sorted(row['knn'],key=row['knn'].get)[:N])

    top_neighbors = []
    for idx in xrange(n):
        topN_sf = sg_dist.get_edges(src_ids=[idx,None],dst_ids=[None,idx]).topk('distance',k=N,reverse=True)
        topN = topN_sf.apply(lambda row: row['__src_id'] if row['__dst_id']==idx else row['__dst_id'])
        top_neighbors.append(topN)

    return gl.SArray(top_neighbors)

####### HACKED TOGETHER NEAREST NEIGHBORS #######


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

#comps = tuple(enumerate(xrange(112312)))
chunksize = int(math.ceil(112312/n_cores))
jobs = tuple(slice_iterable(xrange(112312), chunksize))

def dist_prep(a,feature_arr,feature_arr_2 = None):
    if feature_arr_2 is None:
        artist_features = feature_arr[a]
        def worker(comps):
            return [(ind, JSD(artist_features, feature_arr[ind])) for ind in comps]
    else:
        artist_features_a = feature_arr[a]
        artist_features_b = feature_arr_2[a]
        def worker(comps):
            return [(ind, JSD(artist_features_a, feature_arr[ind]),JSD(artist_features_b, feature_arr_2[ind])) for ind in comps]

    return worker

def calc_neighbors(arrs=1):
    #print 'Running samples'
    pool = mp.Pool(processes=n_cores)
    #start = time.time()
    work_res = pool.map_async(worker, jobs)
    if arrs==1:
        dists = np.array(map(itemgetter(1), sorted(itertools.chain(*work_res.get()))))
    elif arrs==2:
        dists = np.array(map(itemgetter(1,2), sorted(itertools.chain(*work_res.get()))))
    #print "distances calculated in %s" % str(datetime.timedelta(seconds=(time.time()-start)))
    pool.terminate()
    return dists

#################################################

if scrape_data:

    network = pylast.LastFMNetwork(api_key=API_KEY, api_secret=API_SECRET)

    artists = pd.read_table("data/vocab_idx",header=None,names=['artist','idx'])
    all_artist_names = set(artists['artist'])

    done = set()
    try:
        with open(outfile) as fin:
            for line in fin:
                done.add(line[:line.find('\t')])
    except IOError:
        pass


    with open(outfile,'a') as out:
        for i,a in enumerate(artists['artist']):
            print i,a
            while True:
                if a in done:
                    break
                try:
                    artist = network.get_artist(unquote_plus(a))
                    result = artist.get_similar()
                    names = [quote_plus(r.item.name.encode('utf8')).lower() for r in result]
                    print names
                    out.write(a+'\t'+' '.join(names)+'\n')
                    #coverage = len(all_artist_names.intersection(names))
                    break
                except pylast.WSError as e:
                    print e
                    if "The artist you supplied could not be found"==str(e):
                        out.write(a+'\n')
                        break
                except pylast.MalformedResponseError as e:
                    if "Errno 10054" in e.message:
                        print e
                        print "trying again..."
                        time.sleep(30)


### Process Last.fm data:
artist_indices = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    #artist_indices[int(line[1])] = line[0]
    artist_indices[line[0]] = int(line[1])
lastfm_data = {}
with open('lastfm_top_similar_artists') as fin:
    for line in fin:
        try:
            a,top100 = line.strip().split('\t')
        except ValueError:
            lastfm_data[artist_indices[a]] = None
        aid = artist_indices.get(a)
        if aid:
            lastfm_data[aid] = [artist_indices.get(sim) for sim in top100.split()]

lastfm_data = gl.SArray(np.array([lastfm_data.get(i) for i in xrange(112312)])[random_sample])



if calc_nearest_neighbors:

    last = None
    #distance = 'cosine'
    distance = 'JSD'
    distances = {'JSD':JSD}
    out_file = d+'log_knn'

    with open(out_file,'a') as fout:
        for k in k_range:
            print 'K=%s' % k


            # comparison to previous model
            ddir = d+'knn_%s_%s_%s' % (distance,N,k)
            if os.path.exists(ddir):
                topN = gl.SArray(ddir)
            else:
                features = np.load(d+'features_'+str(k)+'.npy')
                if distance in ('cosine',):
                    features = gl.SFrame(features)
                    model=gl.nearest_neighbors.create(dataset=features,distance=distance)
                    result = model.query(features,k=N+1)
                    topN = result[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i]  for i in xrange(1,N+2) if row['knn'][i]!=row['query_label']])
                else:
                    #neigh = NearestNeighbors(n_neighbors=N+1,algorithm='ball_tree',n_jobs=n_cores,metric='pyfunc',func=JSD)
                    #time neigh.fit(features)
                    #time topN = neigh.kneighbors(features[random_sample],return_distance=False)[:,1:]
                    #topN = gl.SArray(topN)

                    #%time topN = triple_apply_knn(features)

                    start = time.time()
                    all_dists = []
                    for i,a in enumerate(random_sample):
                        print "running sample %s/%s\r" % (i+1,sample_size)
                        worker = dist_prep(a,features)
                        dists = calc_neighbors()
                        dists = np.argsort(dists)[:N+1]
                        all_dists.append(dists[dists!=a])
                    topN = gl.SArray(all_dists)
                    print "distances calculated in %s" % str(datetime.timedelta(seconds=(time.time()-start)))

                topN.save(ddir)

            # comparison to last FM
            def proc_fm(row):
                if (row['current'] is None) or (row['fm'] is None):
                    return 0
                else:
                    return len(set(row['current']).intersection(set(row['fm'])))

            combined = gl.SFrame({'current':topN, 'fm':lastfm_data})
            overlap = combined.apply(lambda row: proc_fm(row))
            summary = pd.Series(overlap).describe()
            fout.write('\t'.join(map(str,['fm',k,100]+list(summary)))+'\n')
            fout.flush()


            if last is not None:
                combined = gl.SFrame({'current':topN, 'prev':last})
                overlap = combined.apply(lambda row: len(set(row['current']).intersection(set(row['prev']))))
                summary = pd.Series(overlap).describe()
                fout.write('\t'.join(map(str,['prev',k,N]+list(summary)))+'\n')
                fout.flush()

                overlap_100 = combined.apply(lambda row: len(set(row['current'][:100]).intersection(set(row['prev'][:100]))))
                summary = pd.Series(overlap_100).describe()
                fout.write('\t'.join(map(str,['prev',k,100]+list(summary)))+'\n')
                fout.flush()

            else:
                fout.write('\t'.join(map(str,['prev',k,N]+[0]*8))+'\n')
                fout.write('\t'.join(map(str,['prev',k,100]+[0]*8))+'\n')
                fout.flush()

            # split comparison
            ddir_a = d+'knn_%s_randa_%s_%s' % (distance,N,k)
            ddir_b = d+'knn_%s_randb_%s_%s' % (distance,N,k)
            if os.path.exists(ddir_a) and os.path.exists(ddir_b):
                topN_a = gl.SArray(ddir_a)
                topN_b = gl.SArray(ddir_b)
            else:
                features_a = np.load(d+'features_rand_a_'+str(k)+'.npy')
                features_b = np.load(d+'features_rand_b_'+str(k)+'.npy')
                if distance in ('cosine',):
                    features_a = gl.SFrame(features_a)
                    features_b = gl.SFrame(features_b)
                    model_a=gl.nearest_neighbors.create(dataset=features_a,distance=distance)
                    model_b=gl.nearest_neighbors.create(dataset=features_b,distance=distance)
                    result_a = model_a.query(features_a,k=N+1)
                    result_b = model_b.query(features_b,k=N+1)
                    topN_a = result_a[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i]  for i in xrange(1,N+2) if row['knn'][i]!=row['query_label']])
                    topN_b = result_b[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i]  for i in xrange(1,N+2) if row['knn'][i]!=row['query_label']])
                else:

                    # neigh_a = NearestNeighbors(n_neighbors=N+1,algorithm='ball_tree',n_jobs=n_cores,metric='pyfunc',func=JSD)
                    # neigh_b.fit(features_a)
                    # topN_a = neigh_a.kneighbors(features_a[random_sample],return_distance=False)[:,1:]
                    # topN_a = gl.SArray(topN_a)
                    # neigh_b = NearestNeighbors(n_neighbors=N+1,algorithm='ball_tree',n_jobs=n_cores,metric='pyfunc',func=JSD)
                    # neigh_b.fit(features_b)
                    # topN_b = neigh_b.kneighbors(features_b[random_sample],return_distance=False)[:,1:]
                    # topN_b = gl.SArray(topN_b)

                    #%time topN_a = triple_apply_knn(features_a)
                    #%time topN_b = triple_apply_knn(features_b)
                    start = time.time()
                    all_dists_a = []
                    all_dists_b = []
                    for i,a in enumerate(random_sample):
                        print "running sample %s/%s\r" % (i+1,sample_size)
                        worker = dist_prep(a,features_a,features_b)
                        dists = calc_neighbors(arrs=2)
                        dists_a = np.argsort(dists[:,0])[:N+1]
                        dists_b = np.argsort(dists[:,1])[:N+1]
                        all_dists_a.append(dists_a[dists_a!=a])
                        all_dists_b.append(dists_b[dists_b!=a])
                    topN_a = gl.SArray(all_dists_a)
                    topN_b = gl.SArray(all_dists_b)
                    print "distances calculated in %s" % str(datetime.timedelta(seconds=(time.time()-start)))

                topN_a.save(ddir_a)
                topN_b.save(ddir_b)

            combined = gl.SFrame({'current':topN_a, 'prev':topN_b})

            overlap = combined.apply(lambda row: len(set(row['current']).intersection(set(row['prev']))))
            summary = pd.Series(overlap).describe()
            fout.write('\t'.join(map(str,['split',k,N]+list(summary)))+'\n')
            fout.flush()

            overlap_100 = combined.apply(lambda row: len(set(row['current'][:100]).intersection(set(row['prev'][:100]))))
            summary = pd.Series(overlap_100).describe()
            fout.write('\t'.join(map(str,['split',k,100]+list(summary)))+'\n')
            fout.flush()

            last = topN


