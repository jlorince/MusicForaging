"""
Script to generate random listening sequences with jump distances following known jump distance distribution
"""
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
import logging
from glob import glob
from scipy.spatial.distance import cosine
import sys



jumpdist_path = '/home/jlorince/jumpdists_all'
feature_path =  '/home/jlorince/lda_tests_artists/features_190.npy' # '../GenreModeling/data/features/lda_artists/features_190.npy' #
artist_pop_path = '/home/jlorince/artist_pop'
td_dist_path = '/home/jlorince/scrobble_td.npy'
bins = np.arange(0,1.001,.001)

# setup logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


features = np.load(feature_path)
n_features = features.shape[1]
features = {i:features[i] for i in xrange(len(features))}

def get_features(idx):
    return features.get(idx,np.repeat(np.nan,n_features))

def calc_dist(idx_1,idx_2):
    features1 = get_features(idx_1)
    features2 = get_features(idx_2)
    if np.any(np.isnan(features1)) or np.any(np.isnan(features2)):
        return np.nan
    if np.all(features1==features2):
        return 0.0
    return cosine(features1,features2)

artist_pops = pd.Series([line.split(',')[1] for line in open('/home/jlorince/artist_pop')],dtype=float)
artist_pops.name = 'global_value'

total_users = 145148.
idf = pd.Series(np.log(total_users / np.loadtxt('/home/jlorince/idf_data',dtype=float)))
idf.name = 'idf'

# by-artist cumulative artist count functions
def calc_c_counts(df):
    df['index'] = range(1,len(df)+1)
    return df[['index']]
def calc_c_blockcounts(df):
    df['index'] = df['n'].cumsum()
    return df[['index']]

def return_time_listens(df):
    df['return_time'] = df['overall_index'].shift(-1)-df['overall_index']
    return df[['return_time']]

def return_time(df):
    df['return_time'] = df['overall_index'].shift(-1) - (df['overall_index']+df['n']-1)
    return df[['return_time']]



def calc_values(fi):
    df_raw = pd.read_pickle(fi)
    user = fi.split('/')[-1][:-4]

    # def 1: final prop. listens
    vc = df_raw['artist_idx'].value_counts()
    overall_prop = vc/float(len(df_raw))
    overall_prop.name = 'final_value'
    df_raw = df_raw.join(overall_prop,on='artist_idx')

    # def 3: global popularity

    artist_pops.name = 'global_value'
    df_raw = df_raw.join(artist_pops / artist_pops.sum(),on='artist_idx')

    ### ADD IDF DATA
    df_raw = df_raw.join(idf,on='artist_idx')

    # def 4: tf-idf variant of final prop. listens
    df_raw['final_value_tfidf'] = df_raw['final_value'] * df_raw['idf']

    # copy to generate random shuffle versions
    df_raw_rand = df_raw.copy()
    idx = np.array(df_raw_rand.index)
    np.random.shuffle(idx)
    df_raw_rand = df_raw_rand.reindex(idx).reset_index(drop=True)

    # add jump distances
    df_raw['nextdist'] = df_raw['dist'].shift(-1)
    df_raw_rand['next'] = df_raw_rand['artist_idx'].shift(-1)
    df_raw_rand['nextdist'] = df_raw_rand.apply(lambda row: calc_dist(row['artist_idx'],row['next']),axis=1)

    # generate blocked data
    blocked = df_raw.groupby('block').apply(lambda x: x.iloc[-1])
    blocked['n'] = df_raw.groupby('block')['ts'].count()
    blocked_rand = blocked.copy()
    idx = np.array(blocked_rand.index)
    np.random.shuffle(idx)
    blocked_rand = blocked_rand.reindex(idx).reset_index(drop=True)

    # add jump distancesfor blocks
    blocked_rand['next'] = blocked_rand['artist_idx'].shift(-1)
    blocked_rand['nextdist'] = blocked_rand.apply(lambda row: calc_dist(row['artist_idx'],row['next']),axis=1)

    # def 2: up-to-moment prop. listens

    indices = df_raw.groupby('artist_idx').apply(calc_c_counts)
    df_raw['index'] = indices
    df_raw['overall_index'] = df_raw.index + 1
    df_raw['current_value'] = df_raw['index'] / df_raw['overall_index']

    indices = df_raw_rand.groupby('artist_idx').apply(calc_c_counts)
    df_raw_rand['index'] = indices
    df_raw_rand['overall_index'] = df_raw_rand.index + 1
    df_raw_rand['current_value'] = df_raw_rand['index'] / df_raw_rand['overall_index']

    indices = blocked.groupby('artist_idx').apply(calc_c_blockcounts)
    blocked['index'] = indices
    blocked['overall_index'] = blocked['n'].cumsum()
    blocked['current_value'] = blocked['index'] / blocked['overall_index']

    indices = blocked_rand.groupby('artist_idx').apply(calc_c_blockcounts)
    blocked_rand['index'] = indices
    blocked_rand['overall_index'] = blocked_rand['n'].cumsum()
    blocked_rand['current_value'] = blocked_rand['index'] / blocked_rand['overall_index']

    # def 5: tf-idf variant of up-to-moment prop. listens
    df_raw['current_value_tfidf'] = df_raw['current_value'] * df_raw['idf']
    df_raw_rand['current_value_tfidf'] = df_raw_rand['current_value'] * df_raw_rand['idf']

    blocked['current_value_tfidf'] = blocked['current_value'] * blocked['idf']
    blocked_rand['current_value_tfidf'] = blocked_rand['current_value'] * blocked_rand['idf']

    # add return time data
    rt = df_raw.groupby('artist_idx').apply(return_time_listens)
    df_raw['return_time'] = rt

    rt = df_raw_rand.groupby('artist_idx').apply(return_time_listens)
    df_raw_rand['return_time'] = rt

    rt = blocked.groupby('artist_idx').apply(return_time)
    blocked['return_time'] = rt

    rt = blocked_rand.groupby('artist_idx').apply(return_time)
    blocked_rand['return_time'] = rt

    # SAVE DATA
    with open('/home/jlorince/ee_results/{}'.format(user),'w') as fout:
        def writer(data,basis):
            fout.write(user+'\t'+basis+'\t'+','.join(data.dropna().astype(str))+'\n')
        writer(df_raw.groupby(np.digitize(df_raw['final_value'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_final')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['final_value'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_final_rand')

        writer(df_raw.groupby(np.digitize(df_raw['final_value_tfidf'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_final_tfidf')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['final_value_tfidf'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_final_tfidf_rand')

        writer(df_raw.groupby(np.digitize(df_raw['current_value'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_current')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['current_value'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_current_rand')

        writer(df_raw.groupby(np.digitize(df_raw['current_value_tfidf'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_current_tfidf')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['current_value_tfidf'],bins=bins))['nextdist'].mean(),'scrobbles_nextdist_current_tfidf_rand')

        writer(blocked.groupby(np.digitize(blocked['final_value'],bins=bins))['nextdist'].mean(),'blocks_nextdist_final')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['final_value'],bins=bins))['nextdist'].mean(),'blocks_nextdist_final_rand')

        writer(blocked.groupby(np.digitize(blocked['final_value_tfidf'],bins=bins))['nextdist'].mean(),'blocks_nextdist_final_tfidf')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['final_value_tfidf'],bins=bins))['nextdist'].mean(),'blocks_nextdist_final_tfidf_rand')

        writer(blocked.groupby(np.digitize(blocked['current_value'],bins=bins))['nextdist'].mean(),'blocks_nextdist_current')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['current_value'],bins=bins))['nextdist'].mean(),'blocks_nextdist_current_rand')

        writer(blocked.groupby(np.digitize(blocked['current_value_tfidf'],bins=bins))['nextdist'].mean(),'blocks_nextdist_current_tfidf')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['current_value_tfidf'],bins=bins))['nextdist'].mean(),'blocks_nextdist_current_tfidf_rand')

        writer(df_raw.groupby(np.digitize(df_raw['final_value'],bins=bins))['return_time'].mean(),'scrobbles_return_time_final')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['final_value'],bins=bins))['return_time'].mean(),'scrobbles_return_time_final_rand')

        writer(df_raw.groupby(np.digitize(df_raw['final_value_tfidf'],bins=bins))['return_time'].mean(),'scrobbles_return_time_final_tfidf')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['final_value_tfidf'],bins=bins))['return_time'].mean(),'scrobbles_return_time_final_tfidf_rand')

        writer(df_raw.groupby(np.digitize(df_raw['current_value'],bins=bins))['return_time'].mean(),'scrobbles_return_time_current')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['current_value'],bins=bins))['return_time'].mean(),'scrobbles_return_time_current_rand')

        writer(df_raw.groupby(np.digitize(df_raw['current_value_tfidf'],bins=bins))['return_time'].mean(),'scrobbles_return_time_current_tfidf')
        writer(df_raw_rand.groupby(np.digitize(df_raw_rand['current_value_tfidf'],bins=bins))['return_time'].mean(),'scrobbles_return_time_current_tfidf_rand')

        writer(blocked.groupby(np.digitize(blocked['final_value'],bins=bins))['return_time'].mean(),'blocks_return_time_final')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['final_value'],bins=bins))['return_time'].mean(),'blocks_return_time_final_rand')

        writer(blocked.groupby(np.digitize(blocked['final_value_tfidf'],bins=bins))['return_time'].mean(),'blocks_return_time_final_tfidf')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['final_value_tfidf'],bins=bins))['return_time'].mean(),'blocks_return_time_final_tfidf_rand')

        writer(blocked.groupby(np.digitize(blocked['current_value'],bins=bins))['return_time'].mean(),'blocks_return_time_current')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['current_value'],bins=bins))['return_time'].mean(),'blocks_return_time_current_rand')

        writer(blocked.groupby(np.digitize(blocked['current_value_tfidf'],bins=bins))['return_time'].mean(),'blocks_return_time_current_tfidf')
        writer(blocked_rand.groupby(np.digitize(blocked_rand['current_value_tfidf'],bins=bins))['return_time'].mean(),'blocks_return_time_current_tfidf_rand')






    #cols = ['artist_idx','ts','final_value','final_value_tfidf','current_value','current_value_tfidf','global_value','return_time','nextdist']
    #df_raw[cols].to_pickle('/home/jlorince/values_artists/{}.pkl'.format(user))

    logging.info("User {} processed".format(user))




pool = Pool(cpu_count())
files = glob('/home/jlorince/scrobbles_processed_2_5/*')
donefile = sys.argv[1]
if donefile
    done = set()
    for line in open(donefile):
        if 'processed' in line:
            done.add(line.split()[-2])
    print "{} of {} users already processed".format(len(done),len(files))
files = [f for f in files if f not in done]

#pool.map(calc_values,files)
#pool.close()



