import pandas as pd
from pathos import multiprocessing as mp
import numpy as np
from tqdm import tqdm as tq
from scipy import sparse
from lifelines import NelsonAalenFitter,KaplanMeierFitter
from functools import partial
import sys,math,time,datetime,itertools,os,argparse,dill
from glob import glob


def parse_df(fi,include_time=False):
    if include_time:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],parse_dates=['ts'])
    else:
        return pd.read_table(fi,header=None,names=['song_id','artist_id','ts'],usecols=['song_id','artist_id'])

def gen_exploit_bouts(iden):
    if type(iden)==int:
        df = parse_df('/backup/home/jared/storage/scrobbles-complete{}.txt'.format(uid)).sample(frac=1)
    elif type(iden) in (str,np.string_):
        df = parse_df(iden)
    if (args.min_length is not None) and (len(df)<args.min_length):
        return None
    
    if args.mode == 'encounter':
        #encountered = set()
        encountered = {}
        new = []
        for i,a in enumerate(df.artist_id):
            if args.memory is None:
                if a not in encountered:
                    new.append(1)
                    encountered[a] = None
                else:
                    new.append(0)
            else:
                if a not in encountered:
                    new.append(1)
                else:
                    if i-encountered[a]>args.memory:
                        new.append(1)
                    else:
                        new.append(0)
                encountered[a] = i

        df['explore'] = new

    elif args.mode == 'switch':  
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
    if args.ignore_first is not None:
        df = df[args.ignore_first:]
    exploit_streaks = df[df.explore==0].groupby('explore_block').song_id.count().value_counts().sort_index()
    
    if df.explore.iloc[-1] == 1:
        C = [1]*len(exploit_streaks)
    else:
        C = ([1]*(len(exploit_streaks)-1))+[0]
    return exploit_streaks.values,C

# def survival_curve(uid,mode):
#     result = gen_exploit_bouts(uid,mode=mode)
#     if result==None or len(result[0])==0:
#         return np.full(max_idx,np.nan)
#     T,C = result
#     kmf = KaplanMeierFitter()
#     kmf.fit(T, event_observed=C)
#     return kmf.survival_function_.reindex(range(1,max_idx+1))['KM_estimate'].values

# def hazard_curve(uid,mode,bandwidth=1):
#     result = gen_exploit_bouts(uid,mode=mode)
#     if result==None or len(result[0])==0:
#         return np.full(max_idx,np.nan)
#     T,C = result
#     naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
#     naf.fit(T, event_observed=C)
#     return naf.smoothed_hazard_(bandwidth=bandwidth).reindex(range(1,max_idx+1))['differenced-NA_estimate'].values

# def survival_naive(uid,mode):
#     exploit_streaks = gen_exploit_bouts(uid,mode=mode)
#     if (exploit_streaks is None) or (len(exploit_streaks)==0):
#         return np.full(max_idx,np.nan)
#     cumulative = exploit_streaks[::-1].cumsum()[::-1]
#     result = (cumulative.shift(-1)/cumulative.astype(float)).reindex(range(1,max_idx+1),fill_value=np.nan)
#     return result.values

def concat_TC(files):
    pool = mp.Pool(args.n_procs)
    result = [r for r in pool.map(lambda x: gen_exploit_bouts(x),files) if r is not None]
    T = np.concatenate([x[0] for x in result])
    C = np.concatenate([x[1] for x in result])
    pool.close()
    return T,C

def concat_hazard_curve(T,C):
    naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
    naf.fit(T, event_observed=C)
    #return naf.smoothed_hazard_(bandwidth=bandwidth).reindex(range(1,max_idx+1))['differenced-NA_estimate'].values
    return naf.cumulative_hazard_.reindex(1,args.max_idx+1).values,naf.confidence_interval_.reindex(1,args,max_idx+1).values

def go():
    print args
    T_all,C_all = concat_TC(all_files)
    T_m,C_m = concat_TC(files_m)
    T_f,C_f = concat_TC(files_f)
    for gender,(T,C) in zip(('all','m','f'),((T_all,C_all),(T_m,C_m),(T_f,C_f))):
        naf = NelsonAalenFitter(nelson_aalen_smoothing=False)
        naf.fit(T,event_observed=C)
        dill.dump(naf,open('/backup/home/jared/storage/foraging/cm/{}_{}_shuffle_{}_{}_{}'.format(gender,args.mode,args.min_length,args.ignore_first,args.memory),'wb'))

if __name__ == '__main__':


    parser = argparse.ArgumentParser("Coherence maximizing code")
    parser.add_argument("-m", "--mode",help="",default='encounter',choices=['encounter','switch'])
    parser.add_argument("-i", "--max_idx", help="maximum bout length for which to save results",type=int,default=1000)
    parser.add_argument("-l", "--min_length", help="minimum number of listens for a user",type=int,default=0)
    parser.add_argument("-f", "--ignore_first", help="ignore first n listens from user",type=int,default=0)
    parser.add_argument("-n", "--n_procs", help="",type=int,default=mp.cpu_count())
    parser.add_argument("-y", "--memory", help="memory for encounter-based exploration",default=None)
    #parser.add_argument("-s", "--shuffle", help="memory for encounter-based exploration",default=None) # FIXX THIS
    args = parser.parse_args()

  
    #id_paths = {gender:'S:/UsersData_NoExpiration/jjl2228/foraging/indices_{}'.format(gender) for gender in ('m','f','n')}
    id_paths = {gender:'/backup/home/jared/storage/foraging/cm/indices_{}'.format(gender) for gender in ('m','f','n')}
    gen_ids = False
    for p in id_paths.values():
        if not os.path.exists(p):
            gen_ids = True
            break
    if gen_ids:
        ### METADATA HANDLING
        #user_data = pd.read_table('P:/Projects/BigMusic/jared.rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'],na_values=['\\N'],usecols=['user_id','gender','sample_playcount'])
        user_data = pd.read_table('/backup/home/jared/storage/lastfm_rawdata/lastfm_users.txt',header=None,names=['user_name','user_id','country','age','gender','subscriber','playcount','playlists','bootstrap','registered','type','anno_count','scrobbles_private','scrobbles_recorded','sample_playcount','realname'],na_values=['\\N'],usecols=['user_id','gender','sample_playcount'])

        filtered = user_data.loc[user_data['sample_playcount']>0]

        ids_f = sorted(filtered[filtered['gender']=='f']['user_id'])
        ids_m = sorted(filtered[filtered['gender']=='m']['user_id'])
        ids_n = sorted(filtered[~filtered['gender'].isin(['m','f'])]['user_id'])

    else:
        id_dict = {}
        for gender in ('m','f','n'):
            id_dict[gender] = set([line.strip() for line in open(id_paths[gender])])

    #all_files = glob('p:/Projects/BigMusic/jared.IU/scrobbles-complete/*')
    all_files = glob('/backup/home/jared/storage/scrobbles-complete/*')
    files_m = [f for f in all_files if f[f.rfind('/')+1:-4] in id_dict['m']]
    files_f = [f for f in all_files if f[f.rfind('/')+1:-4] in id_dict['f']]

    ### RUN MAIN PROCESSING - encounter

    # NAIVE
    args.mode = 'encounter'
    args.min_length = 0
    args.ignore_first = 0
    args.memory = None
    go()

    # NAIVE + MEMORY=1000
    args.mode = 'encounter'
    args.min_length = 0
    args.ignore_first = 0
    args.memory = 1000
    go()

    # IGNORE FIRST 5k
    args.mode = 'encounter'
    args.min_length = 10000
    args.ignore_first = 5000
    args.memory = None
    go()

    # IGNORE FIRST 5k + MEMORY=1000
    args.mode = 'encounter'
    args.min_length = 10000
    args.ignore_first = 5000
    args.memory = 1000
    go()

    ### RUN MAIN PROCESSING - switch

    # NAIVE
    args.mode = 'switch'
    args.min_length = 0
    args.ignore_first = 0
    args.memory = None
    go()


    import glob
    for f in glob.glob('/backup/home/jared/storage/foraging/cm/*'):
        if  f.endswith('dill'):
            print f
            current = dill.load(open(f))
            current.cumulative_hazard_.to_csv(f+'_cmhazard')




