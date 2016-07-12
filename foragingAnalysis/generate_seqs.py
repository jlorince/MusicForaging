"""
Script to generate random listening sequences with jump distances following known jump distance distribution
"""
import numpy as np
from scipy.spatial.distance import pdist,squareform
import pandas as pd
from datetime import datetime,timedelta
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
import logging

# number of artists
n_artists = 50000
# sequence length
seq_length = 20000
# number of artifical sequences
n = 10000

jumpdist_path = '/home/jlorince/jumpdists_all'
feature_path =  '/home/jlorince/lda_tests_artists/features_190.npy' # '../GenreModeling/data/features/lda_artists/features_190.npy' #
artist_pop_path = '/home/jlorince/artist_pop'
td_dist_path = '/home/jlorince/scrobble_td.npy'

# setup logging

logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
logging.warning('is when this event was logged.')

# load feature data and generate distance matrix
features = np.load(feature_path)[:n_artists]
logging.info('features loaded')
pw = squareform(pdist(features,metric='cosine'))
logging.info('distance matrix calculated')
pw = np.round(pw,2)
logging.info('distance matrix rounded')


# load overall artist popularity distribution
pops = np.array([int(line.strip().split(',')[1]) for line in open(artist_pop_path)])[:n_artists]
pops = pops / float(pops.sum())
logging.info('features popularity info loaded')

# load time gap distribution
td =np.load(td_dist_path)
logging.info('time distance data loaded')


jump_dist_arr = np.loadtxt(jumpdist_path,delimiter=',',dtype=int)
jump_dist_arr = jump_dist_arr/jump_dist_arr.sum(1,keepdims=True).astype(float)
jumpdist = np.nanmean(jump_dist_arr,0)
logging.info('jump distance data loaded')

def draw(last):
    found = False
    while not found:
        dist = round(np.where(np.random.multinomial(1,pvals=jumpdist)==1)[0][0] / 100.,2)
        candidates = np.where(pw[last]==dist)[0]
        if len(candidates)>0:
            next_idx = np.random.choice(candidates)
            found = True
    return next_idx


def genseq(idx):

    first = np.where(np.random.multinomial(1,pvals=pops)==1)[0][0]
    last = first
    last_ts = datetime.now()
    result = {'artist_idx':[first],'ts':[last_ts]}
    for i in xrange(seq_length-1):
        next_listen = draw(last)
        last = next_listen
        gap_bin = 120*np.where(np.random.multinomial(1,pvals=td)==1)[0][0]
        gap = np.random.randint(gap_bin,gap_bin+120)
        result['artist_idx'].append(next_listen)
        new_ts = last_ts+timedelta(0,gap)
        result['ts'].append(new_ts)
        last_ts = new_ts

    df = pd.DataFrame(result)
    df['block'] = ((df['artist_idx'].shift(1) != df['artist_idx']).astype(int).cumsum())-1
    df.to_pickle(str(idx)+'.pkl')
    logging.info('idx {} complete'.format(idx))

pool = Pool(cpu_count())
indices = range(n)
pool.map(genseq,indices)
pool.close()



