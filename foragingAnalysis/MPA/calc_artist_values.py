"""
Script to generate random listening sequences with jump distances following known jump distance distribution
"""
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
import logging
from glob import glob



jumpdist_path = '/home/jlorince/jumpdists_all'
feature_path =  '/home/jlorince/lda_tests_artists/features_190.npy' # '../GenreModeling/data/features/lda_artists/features_190.npy' #
artist_pop_path = '/home/jlorince/artist_pop'
td_dist_path = '/home/jlorince/scrobble_td.npy'

# setup logging

logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

artist_pops = pd.Series([line.split(',')[1] for line in open('/home/jlorince/artist_pop')],dtype=float)
artist_pops.name = 'global_value'

total_users = 145148.
idf = pd.Series(np.log(total_users / np.loadtxt('/home/jlorince/idf_data',dtype=float)))
idf.name = 'idf'

def calc_values(fi):
    df_raw = pd.read_pickle(fi)
    user = fi.split('/')[-1][:-4]

    # def 1: final prop. listens
    vc = df_raw['artist_idx'].value_counts()
    overall_prop = vc/float(len(df_raw))
    overall_prop.name = 'final_value'
    df_raw = df_raw.join(overall_prop,on='artist_idx')

    # def 2: up-to-moment prop. listens
    def calc_c_counts(df):
        df['index'] = range(1,len(df)+1)
        return df
    df_raw = df_raw.groupby('artist_idx').apply(calc_c_counts)
    df_raw['overall_index'] = df_raw.index + 1
    df_raw['current_value'] = df_raw['index'] / df_raw['overall_index']

    # def 3: global popularity
    df_raw = df_raw.join(artist_pops / artist_pops.sum(),on='artist_idx')

    ### ADD IDF DATA
    df_raw = df_raw.join(idf,on='artist_idx')

    # def 4: tf-idf variant of final prop. listens
    df_raw['final_value_tfidf'] = df_raw['final_value'] * df_raw['idf']

    # def 5: tf-idf variant of up-to-moment prop. listens
    df_raw['current_value_tfidf'] = df_raw['current_value'] * df_raw['idf']

    df_raw.to_pickle('/home/jlorince/values_artists/{}.pkl'.format(user))

    logging.info("User {} processed".format(user))




pool = Pool(cpu_count())
files = glob('/home/jlorince/scrobbles_processed_2_5/*')
pool.map(calc_values,files)
pool.close()



