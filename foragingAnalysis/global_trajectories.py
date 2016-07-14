import numpy as np
#import pathos.multiprocessing as mp
from glob import glob
import pandas as pd

scrobble_path = '/home/jlorince/scrobbles_processed_2_5/*'
#scrobble_path = '/N/dc2/scratch/jlorince/scrobbles_processed_2_5/*'

def choose2(n):
    return n * (n-1) / 2

n=112312
n_choose_2 = choose2(n)

def get_idx(i,j):
    if i>j:
        j,i = i,j
    return n_choose_2 - choose2(n-i) + (j-i-1)

distribution = np.zeros(n_choose_2,dtype=float)

#pool = mp.Pool(process=mp.cpu_count())

files = glob(scrobble_path)


for i,fi in enumerate(files):

    df = pd.read_pickle(fi)
    df = df[['block','artist_idx']].groupby('block').first()
    df['prev'] = df['artist_idx'].shift(1)
    df = df.dropna()
    df['idx'] = df.apply(lambda row: get_idx(row['artist_idx'],row['prev']),axis=1)

    vc = df['idx'].value_counts() #/ float(len(df))

    for idx,cnt in vc.iteritems():
        distribution[int(idx)] += cnt
    print i+1,fi




