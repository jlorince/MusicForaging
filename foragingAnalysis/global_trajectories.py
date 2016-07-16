import numpy as np
#import pathos.multiprocessing as mp
from glob import glob
import pandas as pd
from scipy import sparse

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


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

files = sorted(glob(scrobble_path))


for i,fi in enumerate(files[55075:]):

    df = pd.read_pickle(fi)
    if len(df['artist_idx'].unique())==1:
        continue
    df = df[['block','artist_idx']].groupby('block').first()
    df['prev'] = df['artist_idx'].shift(1)
    df = df.dropna()
    df['idx'] = df.apply(lambda row: get_idx(row['artist_idx'],row['prev']),axis=1)

    vc = df['idx'].value_counts() #/ float(len(df))

    for idx,cnt in vc.iteritems():
        distribution[int(idx)] += cnt
    print i+55075+1,fi


######################################################################
# VIZ ################################################################
######################################################################
import math
import numpy as np
import matplotlib as mpl
import pandas as pd
from scipy.sparse import csr_matrix
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn

n = 112312

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

"""
condensed index -> full matrix index
"""

def calc_row_idx(k, n):
    return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))/2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j

coords = np.loadtxt('output_normed')
sp = load_sparse_csr('global_traj_sparse.npz')
print 'raw data loaded'

coord_data = pd.DataFrame(coords)
coord_data.columns = ['x','y']
df = {'artist_a':[],'artist_b':[],'weight':[]}
for idx,val in zip(sp.indices,sp.data):
    i,j = condensed_to_square(idx,n)
    df['artist_a'].append(i)
    df['artist_b'].append(j)
    df['weight'].append(val)
df = pd.DataFrame(df)
print 'dataframe built'

joined = df.join(coord_data,on='artist_a').join(coord_data,on='artist_b',rsuffix='_b')
joined['weight'] = joined['weight']/joined['weight'].max()
print 'data joined'

fig,ax = plt.subplots(1,1,figsize=(12,12))
df_l = len(joined)
for i,row in enumerate(joined.iterrows()):
    row = row[1]
    ax.plot([row['x'],row['x_b']],[row['y'],row['y_b']],c='blue',alpha=row['weight'],lw=5*row['weight'])
    print "{} / {}".format(i+1,df_l)
fig.savefig('test.pdf',bboc_inches='tight')



