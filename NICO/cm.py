import pandas as pd
import multiprocessing as mp
import numpy as np
from tqdm import tqdm as tq

def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])


def survival(uid):
    df = parse_df('P:/Projects/BigMusic/jared.IU/scrobbles-complete/{}.txt'.format(uid))
    encountered = set()
    new = []
    for a in df.artist_id:
        if a not in encountered:
            new.append(1)
            encountered.add(a)
        else:
            new.append(0)
    df['new'] = new
    df['new_block'] = df['new'].cumsum()
    exploit_streaks = df[df.new==0].groupby('new_block').song_id.count().value_counts().sort_index()
    cumulative = exploit_streaks[::-1].cumsum()[::-1]
    return uid,cumulative.shift(-1)/cumulative.astype(float)

if __name__ == '__main__':

    import sys
    from glob import glob
    import math
    import time,datetime
    import os
    import itertools

    n_procs = mp.cpu_count()
  

    ### METADATA HANDLING
    user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'])

    user_data['sample_playcount'][user_data['sample_playcount']=='\\N'] = 0 
    user_data['sample_playcount'] = user_data['sample_playcount'].astype(int)

    #filtered = user_data.loc[(user_data['gender'].isin(filter_gender)) & (user_data['sample_playcount']>0)][['user_id','gender','sample_playcount']]
    filtered = user_data.loc[user_data['sample_playcount']>0][['user_id','gender','sample_playcount']]

    ids_f = set(filtered[filtered['gender']=='f']['user_id'])
    ids_m = set(filtered[filtered['gender']=='m']['user_id'])
    ids_n = set(filtered[~filtered['gender'].isin(['m','f'])]['user_id'])


    all_files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

    #files_m = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_m],key=os.path.getsize,reverse=True)
    #files_f = sorted([f for f in all_files if int(f[f.rfind('\\')+1:f.rfind('.')]) in ids_f],key=os.path.getsize,reverse=True)

    ### RUN MAIN PROCESSING

    pool = mp.Pool(n_procs)
    
    with open('S:/UsersData_NoExpiration/jjl2228/foraging/cm.txt','w') as out:
        for ids,gender in zip([ids_m,ids_f,ids_n],['m','f','n']):
            for uid,result in tq(pool.imap_unordered(survival,ids,chunksize=100),total=len(ids)):
                result_string = ','.join(result.index.astype(str))+'\t'+','.join(result.values.astype(str))
                out.write("{}\t{}\t{}\n".format(uid,gender,result_string))
    