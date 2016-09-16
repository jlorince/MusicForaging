import pandas as pd
import numpy as np
import os
import glob
import time
import sys
import cPickle
import logging
import datetime
import os


import glob
import pandas as pd
import numpy as np

files = glob.glob('/N/dc2/scratch/jlorince/scrobbles-complete/*')

result = np.zeros(86400)
bins = np.arange(0,86401,1)

for i,f in enumerate(files):
    result += np.histogram(pd.read_table(f,header=None,usecols=[2],names=['ts'],parse_dates=['ts'])['ts'].diff().dropna().apply(lambda x: x.seconds),bins=bins)[0]
    print "{}/{}".format(i+1,len(files))

np.save('/N/u/jlorince/BigRed2/timediff_dist.npy',result)


idx = int(sys.argv[1])
blocksize= 170

inputdir = '/N/dc2/scratch/jlorince/scrobbles-complete/'
outputdir = '/N/dc2/scratch/jlorince/genre_stuff/'
gn_path = '/N/dc2/scratch/jlorince/gracenote_song_data'
#gn_path = '/Users/jaredlorince/Dropbox/Research/misc.data/gracenote_song_data'


now = datetime.datetime.now()
log_filename = now.strftime('genres_%Y%m%d_%H%M%S.log')
logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
rootLogger = logging.getLogger()
# fileHandler = logging.FileHandler(log_filename)
# fileHandler.setFormatter(logFormatter)
# rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)

if os.path.exists(outputdir+'genre_data_'+str(idx)):
    rootLogger.info("Block {} already done".format(idx))
    sys.exit()



offset = idx*blocksize
files = sorted(glob.glob(inputdir+'*'))[offset:offset+blocksize]
if len(files)==0:
    sys.exit()
n_users = len(files)


gn = pd.read_table(gn_path,usecols=['songID','genre1','genre2','genre3']).dropna().set_index('songID')

daterange = pd.date_range(start='2005-07-01',end='2012-12-31',freq='D')

#genres = {'genre1':sorted(gn['genre1'].unique()),'genre2':sorted(gn['genre2'].unique()),'genre3':sorted(gn['genre3'].unique())}
#cPickle.dump(genres,open(outputdir+'gn_genres.pkl','w'))
genres = cPickle.load(open(outputdir+'gn_genres.pkl'))

result = pd.DataFrame(0.,index=daterange,columns=genres['genre1']+genres['genre2']+genres['genre3'])
# if len(done)==0:
#     result = pd.DataFrame(0.,index=daterange,columns=genres['genre1']+genres['genre2']+genres['genre3'])
# else:
#     result = pd.read_pickle(outputdir+'genre_data')

for i,f in enumerate(files):
    user_start = time.time()
    # if f in done:
    #     continue
    df = pd.read_table(f,sep='\t',header=None,names=['item_id','artist_id','scrobble_time'],parse_dates=['scrobble_time']).join(gn,on='item_id',how='left')
    for level in genres:
        vars()['df_'+level] = df.set_index('scrobble_time').groupby([pd.TimeGrouper(freq='D'),level]).count()['item_id'].unstack().reindex(daterange,columns=genres[level])
    concat = pd.concat([df_genre1,df_genre2,df_genre3],axis=1).fillna(0)

    result += concat

    rootLogger.info("{} ({}/{}, {:.1f}, {}, block {})".format(f,i+1,n_users,time.time()-user_start,len(df),idx))
    #time_elapsed = time.time() - start
    # if time_elapsed >= (wall_time-(time_buffer)):
    #     result.to_pickle(outputdir+'genre_data')
    #     sys.exit()

result.to_pickle(outputdir+'genre_data_'+str(idx))


"""

# concat code

import pandas as pd
import datetime
import cPickle
import os

outputdir = '/N/dc2/scratch/jlorince/genre_stuff/'


genres = cPickle.load(open(outputdir+'gn_genres.pkl'))
files = [f for f in os.listdir('.') if f.startswith('genre')]
daterange = pd.date_range(start='2005-07-01',end='2012-12-31',freq='D')
result = pd.DataFrame(0.,index=daterange,columns=genres['genre1']+genres['genre2']+genres['genre3'])
for f in files:
    print f
    result += pd.read_pickle(f)

print result[result.columns[:10]].sum().sum()

"""
