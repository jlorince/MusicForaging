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

class analyze(object):
    def __init__(self,args,logging_level=logging.INFO):

        self.args = args

        # logger setup
        now = datetime.datetime.now()
        log_filename = now.strftime('analysis_%Y%m%d_%H%M%S.log')
        logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
        self.rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(log_filename)
        fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)
        self.rootLogger.setLevel(logging_level)

        self.rootLogger.info("Input arguments: "+str(args))

        self.listen_files = [fi for fi in glob(self.args.pickledir+'*.pkl') if '_patches_' not in fi]
        self.patch_patch = [fi for fi in glob(self.args.pickledir+'*.pkl') if '_patches_' in fi]

    # set up processing pool and run all analyses specified in args
    def run(self):

        self.pool = Pool(self.args.n)
        self.rootLogger.info("Pool started")

        if self.args.jumpdists:

            self.rootLogger.info("Starting jump distance analysis")
            n_bins=10.
            bin_width = 1/n_bins
            bins = np.arange(0,1+bin_width,1/n_bins)
            func_partial = partial(self.artist_jump_distributions,bins=bins,self_jumps=False)
            with open(self.args.resultdir+'jumpdists','w') as fout:
                for user,vals in self.pool.imap(func_partial,self.listen_files):
                    fout.write(user+'\t'+','.join(vals.astype(str))+'\n')

        self.pool.close()
        self.rootLogger.info("Pool closed")

    def artist_jump_distributions(self,fi,bins,self_jumps=False):
        user = fi.split('/')[-1][:-4]
        df = pd.read_pickle(fi)
        if self_jumps:
            vals = np.histogram(df['dist'].dropna(),bins=bins)[0]
        else:
            vals = np.histogram(df['dist'][df['dist']>0],bins=bins)[0]
        return user,vals



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--resultdir", help="specify results location",default='/home/jlorince/results/')
    parser.add_argument("--pickledir", help="specify output dir for pickled dataframes",default='/home/jlorince/scrobbles_processed/')
    parser.add_argument("--jumpdists", help="generate each user's distribution of artist-artist distances",action='store_true')
    parser.add_argument("-n", help="number of processes in processor pool",type=int,default=1)

    args = parser.parse_args()

    mpa = analyze(args,logging_level=logging.INFO)
    mpa.run()

    #python patchAnalyses.py --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ -n 4 --resultdir ./ --jumpdists
