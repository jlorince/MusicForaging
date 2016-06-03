import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

n_trained = 50000


features = np.load('lda_tests_artists/features_190.npy')

dists = pairwise_distances(features)
np.fill_diagonal(dists,0.)

for i in xrange(n_trained,features.shape[0]):
    neighbors = np.argsort(dists[i])
    closest = neighbors[(neighbors!=i)&(neighbors<n_trained)]
    print i,closest dists[i,closest]
