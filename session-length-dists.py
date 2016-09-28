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
outdir = 'P:/Projects/BigMusic/scratch/'

t=30*60

def temporal_threshold(f):
    ser = pd.read_table(f,header=None,usecols=[2],names=['ts'],parse_dates=['ts'])['ts'].diff().dropna().apply(lambda x: x.seconds)
    session_lengths = ((ser>t).cumsum()+1).value_counts()
    if 1 in session_lengths.index:
        session_lengths[1] += 1
    else:
        session_lengths[1] = 1
    result = session_lengths.value_counts()
    result.to_pickle(outdir+f[f.find('\\')+1:])
    return session_lengths.index.max()

def build_hist(f):
    ser = pd.read_pickle(f)
    return ser.reindex(xrange(1,max_length+1)).fillna(0).values




pool = mp.Pool(mp.cpu_count())

result = pool.map(temporal_threshold,files)
max_length = max(result)

temp_files = glob.glob(outdir+'*')

final_result = np.zeros(max_length)
for result in pool.imap_unoredered(build_hist,temp_files):
    final_result += result


with open(outdir+'30min_thresh','w') as fout:
    for i in final_result:
        fout.write(str(i)+'\n')










