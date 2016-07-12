"""
Script to generate random listening sequences with jump distances following known jump distance distribution
"""
import numpy as np
from scipy.spatial.distance import pdist,squareform

# number of artists
n_artists = 10000
jumpdist_path = '../testData/jumpdists_all'
feature_path = '../GenreModeling/data/features/lda_artists/features_190.npy' # 'lda_tests_artists/features_190.npy'
artist_pop_path = '../GenreModeling/data/artist_pop'

# load feature data and generate distance matrix
features = np.load(feature_path)[:n_artists]
pw = squareform(pdist(features,metric='cosine'))
pw = np.round(pw,2)

# load overall artist popularity distribution
pops = np.array([int(line.strip().split(',')[1]) for line in open(artist_pop_path)])
pops = pops / float(pops.sum())


jump_dist_arr = np.loadtxt(jumpdist_path,delimiter=',',dtype=int)
jump_dist_arr = jump_dist_arr/jump_dist_arr.sum(1,keepdims=True).astype(float)
jumpdist = np.nanmean(jump_dist_arr,0)

def draw(last):
    found = False
    while not found:
        dist = round(np.where(np.random.multinomial(1,pvals=jumpdist)==1)[0][0] / 100.,2)
        candidates = np.where(pw[last]==dist)[0]
        if len(candidates)>0:
            next_idx = np.random.choice(candidates)
            found = True
    return next_idx



first = np.where(np.random.multinomial(1,pvals=pops)==1)[0][0]


