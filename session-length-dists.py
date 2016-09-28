import pandas as pd
import numpy as np
import os
import glob
import time
import sys
import cPickle
import logging
import datetime as dt
import multiprocessing as mp


files = glob.glob('P:/Projects/BigMusic/jared.iu/scrobbles-complete/*')
#files = glob.glob(os.path.expanduser('~')+'/scrobbles-complete/*')
outdir = 'P:/Projects/BigMusic/scratch/'
#outdir = os.path.expanduser('~')+'/scratch/'

t=30*60

def temporal_threshold(f):
    ser = pd.read_table(f,header=None,usecols=[2],names=['ts'],parse_dates=['ts'])['ts']
    if len(ser)<1000:
        return 0
    else:
        ser = ser.diff().dropna().apply(lambda x: x.seconds)
        session_lengths = ((ser>t).cumsum()+1).value_counts()
        if 1 in session_lengths.index:
            session_lengths[1] += 1
        else:
            session_lengths[1] = 1
        result = session_lengths.value_counts()
        #result.to_pickle(outdir+f[f.find('\\')+1:])
        result.to_pickle(outdir+f[f.rfind('/')+1:])
        return result.index.max()


def build_hist(f):
    ser = pd.read_pickle(f)
    return ser.reindex(xrange(1,max_length+1)).fillna(0).values



if __name__=='__main__':
    start = time.time()
    pool = mp.Pool(mp.cpu_count())

    result = pool.map(temporal_threshold,files)
    max_length = max(result)
    print "Pickles done in {}, max session length: {}".format(str(datetime.timedelta(seconds=(time.time()-start))),max_length)

    start = time.time()
    temp_files = glob.glob(outdir+'*')
    final_result = np.zeros(max_length)
    for result in pool.imap_unordered(build_hist,temp_files):
        final_result += result
    print "Hists done in {}".format(str(datetime.timedelta(seconds=(time.time()-start))))

    np.save('session_length_dist_30min_1000listens.npy',final_result)
    