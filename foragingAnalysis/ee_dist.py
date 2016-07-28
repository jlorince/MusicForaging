import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
from glob import glob
import time


files_patches = glob('/home/jlorince/patches_clustered/*')
files_scrobbles = glob('/home/jlorince/scrobbles_processed_2_5/*')

def prop_exploit_patches(fi):
    df = pd.read_pickle(fi)
    explore = df[np.isnan(df['patch_clust'])]['n'].sum()
    exploit = df[~np.isnan(df['patch_clust'])]['n'].sum()
    return (explore,exploit)


def prop_exploit_scrobbles(fi):
    blocks = pd.read_pickle(fi)['block']
    cnts = pd.DataFrame({'n':blocks.value_counts().sort_index()})
    cnts['last-n'] = cnts['n'].shift(1)
    cnts['switch'] = cnts.apply(lambda row: 1 if ((row['last-n']==1) and (row['n']>1)) or ((row['last-n']>1) and (row['n']==1)) else 0,axis=1)
    cnts['exp-idx'] = cnts['switch'].cumsum()
    result = cnts.groupby('exp-idx').apply(lambda grp: pd.Series({'n':len(grp),'exploit':0}) if grp['n'].iloc[0]==1 else pd.Series({'n':grp['n'].sum(),'exploit':1}))
    exploit = result[result['exploit']==1]['n'].sum()
    explore = result[result['exploit']==0]['n'].sum()
    return (explore,exploit)



pool = Pool(cpu_count())
start = time.time()
result_patches = pool.map(prop_exploit_patches,files_patches)
print (time.time() - start)/60.
start = time.time()
result_scrobbles = pool.map(prop_exploit_scrobbles,files_scrobbles)
print (time.time() - start)/60.

result_patches = np.vstack(result_patches)
result_scrobbles = np.vstack(result_scrobbles)

np.save('ee_count_patches.npy',result_patches)
np.save('ee_count_scrobbles.npy',result_scrobbles)



