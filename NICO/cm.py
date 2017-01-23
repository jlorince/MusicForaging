import pandas as pd
import multiprocessing as mp
import numpy as np
from tqdm import tqdm as tq
from scipy import sparse
from lifelines import NelsonAalenFitter,KaplanMeierFitter
from functools import partial
import sys,math,time,datetime,itertools,os
from glob import glob


max_idx = 1000
min_length = 0#20000
ignore_first = 0#10000
# mode='encounter'
# bandwidth=1



def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])

def gen_exploit_bouts(uid,mode):
    df = parse_df('P:/Projects/BigMusic/jared.IU/scrobbles-complete/{}.txt'.format(uid))
    if (min_length is not None) and (len(df)<min_length):
        return None
    
    if mode == 'encounter':
        encountered = set()
        new = []
        for a in df.artist_id:
            if a not in encountered:
                new.append(1)
                encountered.add(a)
            else:
                new.append(0)
        df['explore'] = new
    elif mode == 'switch':  
        explore = []
        last = None
        for a in df.artist_id:
            if a == last:
                explore.append(0)
            else:
                explore.append(1)
            last = a
        df['explore'] = explore

    df['explore_block'] = df['explore'].cumsum()
    if ignore_first is not None:
        df = df[ignore_first:]
    exploit_streaks = df[df.explore==0].groupby('explore_block').song_id.count().value_counts().sort_index()
    
    if df.explore.iloc[-1] == 1:
        C = [1]*len(exploit_streaks)
    else:
        C = ([1]*(len(exploit_streaks)-1))+[0]
    return exploit_streaks.values,C


def survival_curve(uid,mode):
    result = gen_exploit_bouts(uid,mode=mode)
    if result==None or len(result[0])==0:
        return np.full(max_idx,np.nan)
    T,C = result
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=C)
    return kmf.survival_function_.reindex(range(1,max_idx+1)).values

def hazard_curve(uid,mode,bandwidth=1):
    result = gen_exploit_bouts(uid,mode=mode)
    if result==None or len(result[0])==0:
        return np.full(max_idx,np.nan)
    naf = NelsonAalenFitter()
    naf.fit(T, event_observed=C)
    return naf.smoothed_hazard_(bandwidth=bandwidth).reindex(range(1,max_idx+1)).values

def survival_naive(uid,mode):
    exploit_streaks = gen_exploit_bouts(uid,mode=mode)
    if (exploit_streaks is None) or (len(exploit_streaks)==0):
        return np.full(max_idx,np.nan)
    cumulative = exploit_streaks[::-1].cumsum()[::-1]
    result = (cumulative.shift(-1)/cumulative.astype(float)).reindex(range(1,max_idx+1),fill_value=np.nan)
    return result.values

if __name__ == '__main__':

    mode = sys.argv[1]
    measure = sys.argv[2]
    if measure == 'hazard':
        func = partial(hazard_curve,mode=mode,bandwidth=sys.argv[3])
    else:
        func = partial(survival_curve,mode=mode)

    n_procs = mp.cpu_count()
  
    id_paths = {gender:'S:/UsersData_NoExpiration/jjl2228/foraging/indices_{}'.format(gender) for gender in ('m','f','n')}
    gen_ids = False
    for p in id_paths.values():
        if not os.path.exists(p):
            gen_ids = True
            break
    if gen_ids:
        ### METADATA HANDLING
        user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'],na_values=['\\N'],usecols=['user_id','gender','sample_playcount'])

        filtered = user_data.loc[user_data['sample_playcount']>0]

        ids_f = sorted(filtered[filtered['gender']=='f']['user_id'])
        ids_m = sorted(filtered[filtered['gender']=='m']['user_id'])
        ids_n = sorted(filtered[~filtered['gender'].isin(['m','f'])]['user_id'])

    else:
        id_dict = {}
        for gender in ('m','f','n'):
            id_dict[gender] = [line.strip() for line in open(id_paths[gender])]

    all_files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')

    ### RUN MAIN PROCESSING

    pool = mp.Pool(n_procs)
    
    for gender,ids in id_dict.iteritems():
        final = []
        if gen_ids:
            out = open(id_paths[gender],'w')
        for uid,result in tq(zip(ids,pool.map(func,ids,chunksize=len(ids)/n_procs)),total=len(ids)):
            final.append(result)
            if gen_ids:
                out.write(str(uid)+'\n')
        if gen_ids:
            out.close()
        final = np.vstack(final)
        #np.save('S:/UsersData_NoExpiration/jjl2228/foraging/cm_{}_{}_{}-{}-{}.npy'.format(mode,gender,max_idx,min_length,ignore_first),final)
        if measure=='hazard':
            np.save('S:/UsersData_NoExpiration/jjl2228/foraging/cm_{}_{}_{}_{}.npy'.format(mode,measure,gender,bandwidth),final)
        else:
            np.save('S:/UsersData_NoExpiration/jjl2228/foraging/cm_{}_{}_{}.npy'.format(mode,measure,gender),final)
    try:
        pool.close()
    except:
        pass
    