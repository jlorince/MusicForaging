import pandas as pd
from glob import glob
from pathos.multiprocessing import ProcessingPool as Pool
import sys
import argparse
from scipy.spatial.distance import cosine,euclidean,pdist
import numpy as np
from itertools import chain,tee, izip, groupby
from functools import partial
import time
import datetime
import logging
import warnings
import setup

class analyze(setup.setup):

    def __init__(self,args,logging_level=logging.INFO):

         super(analyze, self ).__init__(args,logging_level)


    # set up processing pool and run all analyses specified in args
    def run(self):

        self.pool = Pool(self.args.n)
        self.rootLogger.info("Pool started")

        if self.args.jumpdists:

            self.rootLogger.info("Starting jump distance analysis")
            n_bins=100.
            bin_width = 1/n_bins
            bins = np.arange(0,1+bin_width,1/n_bins)
            func_partial = partial(self.artist_jump_distributions,bins=bins,self_jumps=False)
            with open(self.args.resultdir+'jumpdists','w') as fout:
                for user,vals in self.pool.imap(func_partial,self.listen_files):
                    fout.write(user+'\t'+','.join(vals.astype(str))+'\n')

        if self.args.blockdists:
            self.rootLogger.info("Starting block distance analysis")

        self.pool.close()
        self.rootLogger.info("Pool closed")

    def artist_jump_distributions(self,fi,bins,self_jumps=False):
        user = fi.split('/')[-1][:-4]
        df = pd.read_pickle(fi)
        if self_jumps:
            vals = np.histogram(df['dist'].dropna(),bins=bins)[0]
        else:
            vals = np.histogram(df['dist'][df['dist']>0],bins=bins)[0]
        self.rootLogger.info('artist jump distances done for user {} ({})'.format(user,fi))
        return user,vals

    def mean_block_distances(self,fi,n=100):
        user = fi.split('/')[-1][:-4]
        df = pd.read_pickle(fi)
        if 'block' not in df.columns:
            df['block'] = (df['artist_idx'].shift(1) != df['artist_idx']).astype(int).cumsum()
        result = []
        blocks = df[['artist_idx','block']].groupby('block').first()
        for i in xrange(len(blocks)-n):
            first = blocks['artist_id'].iloc[i]
            result.append(np.array(blocks['artist_id'][i+1:i+n+1].apply(lambda val: calc_sim(val,first))))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--resultdir", help="specify results location",default='/home/jlorince/results/')
    parser.add_argument("--pickledir", help="specify output dir for pickled dataframes",default='/home/jlorince/scrobbles_processed/')
    parser.add_argument("-n", help="number of processes in processor pool",type=int,default=1)
    parser.add_argument("--feature_path", help="path to artist feature matrix",default=None) # '/home/jlorince/lda_tests_artists/features_190.npy'

    ### These are all the analyses we can run:
    parser.add_argument("--jumpdists", help="generate each user's distribution of artist-artist distances",action='store_true')
    parser.add_argument("--blockdists", help="",action='store_true')



    args = parser.parse_args()

    mpa = analyze(args,logging_level=logging.INFO)
    mpa.run()

    #python patchAnalyses.py --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ -n 4 --resultdir ./ --jumpdists

    #python patchAnalyses.py --feature_path /Users/jaredlorince/git/MusicForaging/GenreModeling/data/features/lda_artists/features_190.npy --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ -n 4 --resultdir ./ --blockdists
