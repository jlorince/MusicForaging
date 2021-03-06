import pandas as pd
#from glob import glob
#import sys
import argparse
#from scipy.spatial.distance import cosine,euclidean,pdist
import numpy as np
#from itertools import chain,tee, izip, groupby
#from functools import partial
#import time
import datetime
import logging
#import warnings
from scipy import sparse
#import os



#### cat $(wc -l * | grep '198\s' | gawk '{print $2}')  > patch_len_dists_concat


class setup(object):

    # init just takes in command line arguments and sets up logging
    def __init__(self,args,logging_level=logging.INFO):

        self.args = args

        # logger setup
        now = datetime.datetime.now()
        log_filename = now.strftime('setup_%Y%m%d_%H%M%S.log')
        logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
        self.rootLogger = logging.getLogger()
        #fileHandler = logging.FileHandler(log_filename)
        #fileHandler.setFormatter(logFormatter)
        #self.rootLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)
        self.rootLogger.setLevel(logging_level)

        #self.rootLogger.info("Input arguments: "+str(args))

        if self.args.feature_path:
            features = np.load(self.args.feature_path)
            self.n_features = features.shape[1]
            self.features = {i:features[i] for i in xrange(len(features))}

    @staticmethod
    def userFromFile(fi):
        return fi.split('/')[-1].split('_')[-1][:-4]


    # set up processing pool and run all analyses specified in args
    def run(self):

        if self.args.preprocess:
            #self.rootLogger.info("Starting preprocessing")
            self.preprocess()
            #self.rootLogger.info("Preprocessing complete")

        if self.args.patch_basis is not None:
            #self.rootLogger.info("Starting patch summaries")
            self.summarize_patches()
            #self.rootLogger.info("Patch summaries complete")

        if self.args.blockdists:
            #self.rootLogger.info("Starting block distance analysis")
            self.mean_block_distances(self.args.file)

        if self.args.blockgaps:
            #self.rootLogger.info("Starting block distance analysis")
            self.blockgaps(self.args.file)

        if self.args.scrobblegaps:
            #self.rootLogger.info("Starting block distance analysis")
            self.scrobble_gaps(self.args.file)

        if self.args.ee_artists:
            self.ee_artists(self.args.file)

        if self.args.ee_artists_2:
            self.ee_artists_2(self.args.file)

        if self.args.ee_artists_dists:
            self.ee_artists_dists(self.args.file)

        if self.args.block_len_dists:
            self.block_len_dists(self.args.file)




    # Calls preprocessing code to load raw text files and convert to dataframes, adding features, disances, etc.
    def preprocess(self):

        self.artist_idx_feature_map = {}
        for line in open(self.args.suppdir+'artist_idx_feature_map'):
            k,v = line.strip().split('\t')
            self.artist_idx_feature_map[float(k)] = int(v)

        if self.args.file:
            result = self.processor(fi=self.args.file,output_dir=self.args.pickledir,is_sorted=True,features=self.features,dist=self.args.distance_metric,session_threshold=self.args.session_thresh,dist_threshold=self.args.dist_thresh, min_patch_length=self.args.min_patch_length,artist_idx_feature_map=self.artist_idx_feature_map)

                # if self.args.patch_len_dist:
                #     user,vals_simple,vals_shuffle = result
                #     with open(self.args.resultdir+user,'a') as fout:
                #         if vals_simple is not None:
                #             fout.write('\t'.join([user,'simple',str(self.args.dist_thresh)])+'\t'+','.join(vals_simple.astype(str))+'\n')
                #         fout.write('\t'.join([user,'shuffle',str(self.args.dist_thresh),str(self.args.min_patch_length)])+'\t'+','.join(vals_shuffle.astype(str))+'\n')


        else:
            if args.rawtext:
                if self.args.skip_complete:
                    done =  set([self.userFromFile(fi) for fi in glob(self.args.pickledir+'*.pkl') if '_patches_' not in fi and fi.startswith(self.args.prefix_output)])
                else:
                    done = set()
                files = [fi for fi in glob(self.args.datadir+'*.txt') if self.userFromFile(fi) not in done]
            else:
                if self.args.skip_complete:
                    done =  set([self.userFromFile(fi) for fi in glob(self.args.pickledir+'*.pkl') if '_patches_' not in fi and fi.startswith(self.args.prefix_output)])
                else:
                    done = set()
                files = [fi for fi in glob(self.args.pickledir+'*.pkl') if '_patches_' not in fi and fi.startswith(self.args.prefix_input) and self.userFromFile(fi) not in done]

            self.n_files = len(files)

            self.rootLogger.debug(files)

            func_partial = partial(self.processor,output_dir=self.args.pickledir,is_sorted=True,features=self.features,dist=self.args.distance_metric,session_threshold=self.args.session_thresh,dist_threshold=self.args.dist_thresh, min_patch_length=self.args.min_patch_length,artist_idx_feature_map=self.artist_idx_feature_map)

            self.pool = Pool(self.args.n)
            self.rootLogger.info("Pool started")
            self.pool.map(func_partial,files)
            self.pool.close()
            self.rootLogger.info("Pool closed")

    # Jensen Shannon Distance (Sqrt of Jensen Shannon Divergence)
    @staticmethod
    def JSD(P, Q):
        if np.all(np.isnan(P)) or np.all(np.isnan(Q)):
            return np.nan
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return np.sqrt(np.clip(0.5 * (entropy(_P, _M) + entropy(_Q, _M)),0,1))

    # Calculate distance between any two feature arrays
    def calc_dist(self,idx_1,idx_2,metric='cosine'):
        features1 = self.get_features(idx_1)
        features2 = self.get_features(idx_2)
        if np.any(np.isnan(features1)) or np.any(np.isnan(features2)):
            return np.nan
        if np.all(features1==features2):
            return 0.0
        if metric == 'JSD':
            return self.JSD(features1,features2)
        elif metric == 'cosine':
            return cosine(features1,features2)
        elif metric == 'euclidean':
            return euclidean(features1,features2)


    # "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    @staticmethod
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    # segment patch, generating both simple and shuffle-based indices
    def patch_segmenter(self,df,metric,min_length,dist_thresh):
        l = df['artist_idx']
        indices = list(np.array([len(list(v)) for g,v in groupby(l)][:-1]).cumsum())
        new_indices = []

        for b in indices:
            dist = self.calc_dist(df.iloc[b]['artist_idx'],df.iloc[b-1]['artist_idx'], metric=metric)

            if (np.isnan(dist)) or (dist >= dist_thresh):
                new_indices.append(b)

        if new_indices:

            last_patch = False
            final_indices = []
            for i,(a,b) in enumerate(self.pairwise([0]+new_indices+[len(df)])):
                if b-a>=min_length:
                    if a>0:
                        final_indices.append(a)
                    last_patch = True
                else:
                    if last_patch:
                        final_indices.append(a)
                    last_patch = False

            return final_indices,new_indices

        return new_indices,new_indices

    # retrieve features from feature matrix, given an artist idx. Return array of np.nans if artist idx is null
    def get_features(self,idx):
        return self.features.get(idx,np.repeat(np.nan,self.n_features))
        # if np.isnan(idx):
        #     return np.repeat(np.nan,self.features.shape[1])
        # else:
        #     return self.features[int(idx)]


    # Core preprocessing code. Can take in raw text files, or pickle files (in which case feature/dist values are updated appropriately)
    def processor(self,fi,output_dir,is_sorted=True,features=None,dist='cosine',session_threshold=None,dist_threshold=0.2,min_patch_length=5,artist_idx_feature_map=None):

        # get user_id from filename
        user = self.userFromFile(fi)
        self.rootLogger.debug('processor called (user {})'.format(user))

        if fi.endswith('.txt'):

            if output_dir is None:
                raise("output path must be specified!")
            if artist_idx_feature_map is None:
                raise("artist_idx_feature_map_path must be provided!")

            df = pd.read_table(fi,
                    header=None,
                    names=['artist_id','ts'],
                    parse_dates=['ts'])
            if not is_sorted:
                df = df.sort_values(by='ts')

            df['td'] = df['ts']-df.shift(1)['ts']
            df['td'] = df['td'].astype(int) / 10**9
            df['artist_idx'] = df['artist_id'].apply(lambda x: artist_idx_feature_map.get(x))
            n = float(len(df))
            n_null = df['artist_idx'].isnull().sum()
            notnull = n-n_null
            propnull = n_null/n
            if notnull<1000 or (propnull >= 0.05):
                self.rootLogger.info('User {} SKIPPED ({} non null, {:.1f}% null) ({})'.format(user, notnull,100*propnull,fi))
                return None
            self.rootLogger.debug('DF loaded (user {})'.format(user))

        elif fi.endswith('.pkl'):
            df = pd.read_pickle(fi)

        # get features and calculate distances
        if features is not None:
            #df['features'] = df['artist_idx'].apply(lambda idx: self.get_features(idx))
            #df['features_shift'] = df['features'].shift(1)
            df['prev'] = df['artist_idx'].shift(1)

            df['dist'] = df.apply(lambda row: self.calc_dist(row['artist_idx'],row['prev'],metric=dist),axis=1)

            self.rootLogger.debug('features and dists done (user {})'.format(user))

        if session_threshold == 0:
            df['session'] = 0

        elif (session_threshold is not None) and (session_threshold>0):
            if 'td' not in df.columns:
                df['td'] = df['ts']-df.shift(1)['ts']
                df['td'] = df['td'].astype(int) / 10**9
            session_idx = 0
            session_indices = []
            for val in df['td']>=session_threshold:
                if val:
                    session_idx +=1
                session_indices.append(session_idx)
            df['session'] = session_indices
            self.rootLogger.debug('session indices done (user {})'.format(user))



        if (min_patch_length is not None) and (dist_threshold is not None):

            self.rootLogger.debug('starting patch segmentation for user {})'.format(user))

            indices_shuffle = np.zeros(len(df),dtype=int)
            indices_simple = np.zeros(len(df),dtype=int)
            offset_shuffle = 0
            idx_shuffle=0
            offset_simple = 0
            idx_simple=0

            ### NEED TO REWORK THIS BIT TO LOSE SOME REDUNDANCY

            for session in df.groupby('session'):
                result_shuffle,result_simple = self.patch_segmenter(session[1],metric=dist,min_length=min_patch_length,dist_thresh=dist_threshold)
                #if session[0]==0:
                #    print result_shuffle,result_simple
                #    sys.exit()
                n=len(session[1])

                if len(result_shuffle)==0:
                    indices_shuffle[offset_shuffle:offset_shuffle+n] = idx_shuffle
                    idx_shuffle+=1
                else:
                    indices_shuffle[offset_shuffle:offset_shuffle+result_shuffle[0]] = idx_shuffle
                    idx_shuffle+=1
                    for v, w in self.pairwise(result_shuffle):
                        indices_shuffle[offset_shuffle+v:offset_shuffle+w] = idx_shuffle
                        idx_shuffle+=1
                    indices_shuffle[offset_shuffle+result_shuffle[-1]:offset_shuffle+result_shuffle[-1]+n] = idx_shuffle
                    idx_shuffle+=1
                offset_shuffle += n

                if len(result_simple)==0:
                    indices_simple[offset_simple:offset_simple+n] = idx_simple
                    idx_simple+=1
                else:
                    indices_simple[offset_simple:offset_simple+result_simple[0]] = idx_simple
                    idx_simple+=1
                    for v, w in self.pairwise(result_simple):
                        indices_simple[offset_simple+v:offset_simple+w] = idx_simple
                        idx_simple+=1
                    indices_simple[offset_simple+result_simple[-1]:offset_simple+result_simple[-1]+n] = idx_simple
                    idx_simple+=1
                offset_simple += n

            if result_shuffle:
                indices_shuffle[offset_shuffle+result_shuffle[-1]:] = idx_shuffle
            else:
                indices_shuffle[offset_shuffle:] = idx_shuffle

            if result_simple:
                indices_simple[offset_simple+result_simple[-1]:] = idx_simple
            else:
                indices_simple[offset_simple:] = idx_simple

            df['patch_idx_shuffle'] = indices_shuffle
            df['patch_idx_simple'] = indices_simple

            self.rootLogger.debug('patch indices done (user {})'.format(user))

            # add artist block info
            ### https://stackoverflow.com/questions/14358567/finding-consecutive-segments-in-a-pandas-data-frame
            ### -1 for zero-based indexing

            df['block'] = ((df['artist_idx'].shift(1) != df['artist_idx']).astype(int).cumsum())-1

            self.rootLogger.debug('artist blocks done (user {})'.format(user))

        cols = ['ts','artist_idx','dist','session','patch_idx_shuffle','patch_idx_simple','block']

        df = df[list(set(df.columns).intersection(cols))]
        if self.args.save:
            df.to_pickle('{}{}.pkl'.format(output_dir,user))

        if self.args.patch_len_dist:
            self.patch_length_distributions(user,df,bins=np.arange(0,1001,1),method=self.args.patch_len_dist)
            #return user,vals_simple,vals_shuffle


        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))
        return None

    # calculate patch summary measures (mean feature array, diversity, etc.). Applied to each patch
    def patch_measures(self,df,agg_stats=True,metric='cosine'):
        first = df.iloc[0]
        n = len(df)
        start = first['ts']
        if agg_stats:
            #artists = df['artist_idx'].values
            if (n==1) or (len(df['artist_idx'].unique())==1):
                diversity = 0.
                centroid = first['features']
            else:
                features = np.array([f for f in df['features']])
                # I expect to see RuntimeWarnings in this block
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    centroid = np.nanmean(features,axis=0)
                    diversity = np.nanmean(pdist(features,metric=metric))

            #return pd.Series({'diversity':diversity,'centroid':centroid,'start_ts':start,'n':n,'artists':artists})
            return pd.Series({'diversity':diversity,'centroid':centroid,'start_ts':start,'n':n})

    # generate patch summary for each user, and save resulting pickle
    def patch_summary(self,fi,basis,metric):
        user = self.userFromFile(fi)
        df = pd.read_pickle(fi)
        df['features'] = df['artist_idx'].apply(lambda idx: self.get_features(idx))
        if basis=='block':
            agg_stats=False
        elif basis in ('patch_idx_shuffle','patch_idx_simple'):
            agg_stats = True
        else:
            raise("Invalid patch basis")
        result = df.groupby(basis).apply(self.patch_measures,agg_stats,metric)
        #result['start_idx'] = result['n'].cumsum().shift(1).fillna(0).astype(int)
        result.reset_index(drop=True).to_pickle('{}{}_patches_{}.pkl'.format(self.args.resultdir,user,basis))
        self.rootLogger.info('Patches processed for user {} successfully ({})'.format(user,fi))

    # run patch summaries for all users
    def summarize_patches(self):

        if self.args.file:
            self.patch_summary(fi=self.args.file,basis=self.args.patch_basis,metric=self.args.distance_metric)

        else:

            if self.args.skip_complete:
                done =  set([self.userFromFile(fi) for fi in glob(self.args.pickledir+'*.pkl') if '_patches_' in fi and fi.startswith(self.args.prefix_output)])
            else:
                done = set()
            files = [fi for fi in glob(self.args.pickledir+'*.pkl') if '_patches_' not in fi and fi.startswith(self.args.prefix_input) and self.userFromFile(fi) not in done]
            func_partial = partial(self.patch_summary,basis=self.args.patch_basis,metric=self.args.distance_metric)
            self.rootLogger.info("Pool started")
            self.pool.map(func_partial,files)
            self.pool.close()
            self.rootLogger.info("Pool closed")


    def patch_length_distributions(self,user,df,bins,method):
        n_listens = float(len(df))
        if self.args.min_patch_length==2:
            vc_simple = df['patch_idx_simple'].value_counts().values
            counts_simple = np.clip(vc_simple,0,1000)
            vals_simple = np.histogram(counts_simple,bins=bins)[0]
            listens_simple = np.array([i*c for i,c in enumerate(vals_simple)])
            listens_simple[-1] = vc_simple[vc_simple>=1000].sum()
            listens_simple = listens_simple / n_listens

            vc_block = df['block'].value_counts().values
            counts_block = np.clip(vc_block,0,1000)
            vals_block = np.histogram(counts_block,bins=bins)[0]
            listens_block = np.array([i*c for i,c in enumerate(vals_block)])
            listens_block[-1] = vc_block[vc_block>=1000].sum()
            listens_block = listens_block / n_listens

            with open(self.args.resultdir+user,'a') as fout:
                fout.write('\t'.join([user,'block','patches',str(self.args.dist_thresh),str(self.args.min_patch_length)])+'\t'+','.join(vals_block.astype(str))+'\n')
                fout.write('\t'.join([user,'block','listens',str(self.args.dist_thresh),str(self.args.min_patch_length)])+'\t'+','.join(listens_block.astype(str))+'\n')
                fout.write('\t'.join([user,'simple','patches',str(self.args.dist_thresh),str(self.args.min_patch_length)])+'\t'+','.join(vals_simple.astype(str))+'\n')
                fout.write('\t'.join([user,'simple','listens',str(self.args.dist_thresh),str(self.args.min_patch_length)])+'\t'+','.join(listens_simple.astype(str))+'\n')

        vc_shuffle = df['patch_idx_shuffle'].value_counts().values
        counts_shuffle = np.clip(vc_shuffle,0,1000)
        vals_shuffle = np.histogram(counts_shuffle,bins=bins)[0]
        listens_shuffle = np.array([i*c for i,c in enumerate(vals_shuffle)])
        listens_shuffle[-1] = vc_shuffle[vc_shuffle>=1000].sum()
        listens_shuffle = listens_shuffle / n_listens


        with open(self.args.resultdir+user,'a') as fout:
            fout.write('\t'.join([user,'shuffle','patches',str(self.args.dist_thresh),str(self.args.min_patch_length)])+'\t'+','.join(vals_shuffle.astype(str))+'\n')
            fout.write('\t'.join([user,'shuffle','listens',str(self.args.dist_thresh),str(self.args.min_patch_length)])+'\t'+','.join(listens_shuffle.astype(str))+'\n')

    def mean_block_distances(self,fi,n=100,shuffle=False):


        def hash_handler(a,frst):
            if frst>a:
                frst,a = a,frst
            if frst not in dhash:
                dhash[frst] = {}
                result = self.calc_dist(frst,a)
                dhash[frst][a] = result
            else:
                result = dhash[frst].get(a)
                if result is None:
                    result = self.calc_dist(frst,a)
                    dhash[frst][a] = result
            return result

        def cos_nan(arr1,arr2):
            if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
                return np.nan
            else:
                return cosine(arr1,arr2)

        user = fi.split('/')[-1][:-4]
        df = pd.read_pickle(fi)


        if os.path.exists(self.args.resultdir+user):
            levels = {'scrobble':False,'block':False,'D':True,'W':False,'M':False}
        else:
            levels = {'scrobble':True,'block':True,'D':True,'W':True,'M':True}

        if levels['scrobble']:
            result = []
            dhash = {}
            if shuffle:
                blocks = df.copy()
                idx = np.array(blocks.index)
                np.random.shuffle(idx)
                blocks = blocks.reindex(idx)
            blocks = df.copy()
            for i in xrange(len(blocks)-n):
                first = blocks['artist_idx'].iloc[i]
                result.append(np.array(df['artist_idx'][i+1:i+n+1].apply(lambda val: hash_handler(val,first))))
            result = np.nanmean(np.vstack(result),0)
            with open(self.args.resultdir+user,'a') as fout:
                fout.write('\t'.join([user,'scrobble',','.join(result.astype(str))])+'\n')

        if levels['block']:
            result = []
            blocks = df[['artist_idx','block']].groupby('block').first()
            if shuffle:
                idx = np.array(blocks.index)
                np.random.shuffle(idx)
                blocks = blocks.reindex(idx)
            for i in xrange(len(blocks)-n):
                first = blocks['artist_idx'].iloc[i]
                result.append(np.array(blocks['artist_idx'][i+1:i+101].apply(lambda val: hash_handler(val,first))))
            result = np.nanmean(np.vstack(result),0)
            with open(self.args.resultdir+user,'a') as fout:
                fout.write('\t'.join([user,'block',','.join(result.astype(str))])+'\n')


        df['features'] = df['artist_idx'].apply(lambda idx: self.get_features(idx))
        df = df.set_index('ts')['features']

        for res,n in (('D',365),('W',52),('M',12)):
            if levels[res]:
                result = []
                blocks = df.resample(res).aggregate(lambda ser: np.nanmean(np.vstack(ser.values),axis=0) if len(ser)>0 else np.repeat(np.nan,self.n_features))
                if shuffle:
                    idx = np.array(blocks.index)
                    np.random.shuffle(idx)
                    blocks = blocks.reindex(idx)
                for i in xrange(len(blocks)-n):
                    first = blocks.iloc[i]
                    result.append(np.array(blocks[i+1:i+n+1].apply(lambda val: cos_nan(val,first))))
                result = np.nanmean(np.vstack(result),0)
                with open(self.args.resultdir+user,'a') as fout:
                        fout.write('\t'.join([user,res,','.join(result.astype(str))])+'\n')

        self.rootLogger.info('Block distances for user {} processed successfully ({})'.format(user,fi))

    def blockgaps(self,fi):
        user = self.userFromFile(fi)
        result = []
        df = pd.read_pickle(fi)[['ts','artist_idx','block']].groupby('block').first()
        bins = np.arange(0,31,1)
        day = np.timedelta64(1, 'D')
        for artist in df['artist_idx'].dropna().unique():
            current = df[df['artist_idx']==artist]['ts']
            td = ((current-current.shift(1)).dropna())/day
            vals = np.histogram(td,bins=bins)[0]
            result.append(vals/float(vals.sum()))
        result = np.nanmean(np.vstack(result),0)
        with open(self.args.resultdir+user,'w') as fout:
            fout.write('\t'.join([user,','.join(result.astype(str))])+'\n')
        self.rootLogger.info('Gap times for user {} processed successfully ({})'.format(user,fi))

    def scrobble_gaps(self,fi):
        user = self.userFromFile(fi)
        result = []
        df = pd.read_pickle(fi)['ts']
        bins = np.arange(0,60 * 60 * 24 * 30,120)
        td = (df-df.shift(1)).dropna().apply(lambda x: x.total_seconds())
        vals = np.histogram(td,bins=bins)[0]
        result = vals/float(vals.sum())
        with open(self.args.resultdir+user,'w') as fout:
            fout.write('\t'.join([user,','.join(result.astype(str))])+'\n')
        self.rootLogger.info('Gap times for user {} processed successfully ({})'.format(user,fi))

    def ee_artists(self,fi):
        user = self.userFromFile(fi)
        blocks = pd.read_pickle(fi)['block']
        result = blocks.value_counts().value_counts()
        arr = result.reindex(xrange(1,max(result.index)+1),fill_value=0.).values
        final_result = arr/(np.cumsum(arr[::-1])[::-1])
        final_result = sparse.csr_matrix(final_result)

        with open(self.args.resultdir+user,'w') as fout:
            fout.write(user+'\t'+':'.join([','.join(a.astype(str)) for a in final_result.data,final_result.indices,final_result.indptr])+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))

    def ee_artists_2(self,fi):
        user = self.userFromFile(fi)
        blocks = pd.read_pickle(fi)['block']
        cnts = pd.DataFrame({'n':blocks.value_counts().sort_index()})
        cnts['last-n'] = cnts['n'].shift(1)
        cnts['switch'] = cnts.apply(lambda row: 1 if ((row['last-n']==1) and (row['n']>1)) or ((row['last-n']>1) and (row['n']==1)) else 0,axis=1)
        cnts['exp-idx'] = cnts['switch'].cumsum()
        result = cnts.groupby('exp-idx').apply(lambda grp: pd.Series({'n':len(grp),'exploit':0}) if grp['n'].iloc[0]==1 else pd.Series({'n':grp['n'].sum(),'exploit':1}))[:-1]
        #result = cnts.groupby('exp-idx').apply(lambda grp: pd.Series({'n':len(grp),'exploit':0}) if grp['n'].iloc[0]==1 else pd.Series({'n':grp['n']iloc[-1],'exploit':1}))[:-1]
        arr_exploit = result[result['exploit']==1]['n'].value_counts()
        arr_exploit = arr_exploit.reindex(xrange(1,max(arr_exploit.index)+1),fill_value=0.).values
        arr_explore = result[result['exploit']==0]['n'].value_counts()
        arr_explore = arr_explore.reindex(xrange(1,max(arr_explore.index)+1),fill_value=0.).values

        final_result_exploit = arr_exploit/(np.cumsum(arr_exploit[::-1])[::-1])
        final_result_exploit = sparse.csr_matrix(final_result_exploit)

        final_result_explore = arr_explore/(np.cumsum(arr_explore[::-1])[::-1])
        final_result_explore = sparse.csr_matrix(final_result_explore)

        with open(self.args.resultdir+user,'w') as fout:
            fout.write(user+'\t'+'explore'+'\t'+':'.join([','.join(a.astype(str)) for a in final_result_explore.data,final_result_explore.indices,final_result_explore.indptr])+'\n')
            fout.write(user+'\t'+'exploit'+'\t'+':'.join([','.join(a.astype(str)) for a in final_result_exploit.data,final_result_exploit.indices,final_result_exploit.indptr])+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))

    def ee_artists_dists(self,fi):
        user = self.userFromFile(fi)
        blocks = pd.read_pickle(fi)['block']
        cnts = pd.DataFrame({'n':blocks.value_counts().sort_index()})
        cnts['last-n'] = cnts['n'].shift(1)
        cnts['switch'] = cnts.apply(lambda row: 1 if ((row['last-n']==1) and (row['n']>1)) or ((row['last-n']>1) and (row['n']==1)) else 0,axis=1)
        cnts['exp-idx'] = cnts['switch'].cumsum()
        result = cnts.groupby('exp-idx').apply(lambda grp: pd.Series({'n':len(grp),'exploit':0}) if grp['n'].iloc[0]==1 else pd.Series({'n':grp['n'].sum(),'exploit':1}))[:-1]
        #result = cnts.groupby('exp-idx').apply(lambda grp: pd.Series({'n':len(grp),'exploit':0}) if grp['n'].iloc[0]==1 else pd.Series({'n':grp['n']iloc[-1],'exploit':1}))[:-1]
        arr_exploit = result[result['exploit']==1]['n'].value_counts()
        arr_exploit = sparse.csr_matrix(arr_exploit.reindex(xrange(1,max(arr_exploit.index)+1),fill_value=0.).values)
        arr_explore = result[result['exploit']==0]['n'].value_counts()
        arr_explore = sparse.csr_matrix(arr_explore.reindex(xrange(1,max(arr_explore.index)+1),fill_value=0.).values)

        with open(self.args.resultdir+user,'w') as fout:
            fout.write(user+'\t'+'explore'+'\t'+':'.join([','.join(a.astype(str)) for a in arr_explore.data,arr_explore.indices,arr_explore.indptr])+'\n')
            fout.write(user+'\t'+'exploit'+'\t'+':'.join([','.join(a.astype(str)) for a in arr_exploit.data,arr_exploit.indices,arr_exploit.indptr])+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))


    def block_len_dists(self,fi):
        user = self.userFromFile(fi)
        blocks = pd.read_pickle(fi)['block']
        result = blocks.value_counts().value_counts()
        arr = result.reindex(xrange(1,max(result.index)+1),fill_value=0.).values
        final_result = sparse.csr_matrix(arr)
        with open(self.args.resultdir+user,'w') as fout:
            fout.write(user+'\t'+':'.join([','.join(a.astype(str)) for a in final_result.data,final_result.indices,final_result.indptr])+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))






if __name__ == '__main__':

    parser = argparse.ArgumentParser("Need to add some more documentation")

    parser.add_argument("-f", "--file",help="If provided, run setup for this file only",default=None,type=str)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",action="store_true")
    parser.add_argument("-p", "--preprocess", help="perform preprocessing of listening histories",action="store_true")
    parser.add_argument("-r", "--rawtext",help="Load scrobbles from raw text files. If not specififed, assumes files are already pickled and saved in `pickledir`",action="store_true")
    parser.add_argument("-s","--save",help='save newly generated DFs',action="store_true")
    parser.add_argument("--pickledir", help="specify output dir for pickled dataframes",default='/home/jlorince/scrobbles_processed/')
    parser.add_argument("--datadir", help="specify base directory containing input files",default='/home/jlorince/scrobbles/')
    parser.add_argument("--suppdir", help="specify supplementary data location",default='/home/jlorince/support/')
    parser.add_argument("--resultdir", help="specify results location",default='/home/jlorince/results/')
    parser.add_argument("--session_thresh", help="session segmentation threshold. Use 0 for no time-based segmentation.",type=int,default=None) # 1800
    parser.add_argument("--min_patch_length", help="minimum patch length",type=int,default=None) # 5
    parser.add_argument("--dist_thresh", help="distance threshold defining patch neigborhood",type=float,default=None) # 0.2
    parser.add_argument("-n", help="number of processes in processor pool",type=int,default=1)
    parser.add_argument("--feature_path", help="path to artist feature matrix",default=None) # '/home/jlorince/lda_tests_artists/features_190.numpy'
    parser.add_argument("--distance_metric", help="distance metric",type=str,default='cosine')
    parser.add_argument("--patch_basis", help="If specified, perform patch summaries with the given basis",type=str,choices=['block','patch_idx_shuffle','patch_index_simple'])
    parser.add_argument("--skip_complete", help="If specified, check for existing files and skip if they exist",action='store_true')
    parser.add_argument("--prefix_input", help="inpout file prefix",type=str,default='')
    parser.add_argument("--prefix_output", help="output file prefix",type=str,default='')
    #parser.add_argument("--patch_len_dist", help="compute distribution of patch lengths",default=None,type=str,choices=['shuffle','simple','block','both'])
    parser.add_argument("--patch_len_dist", help="compute distribution of patch lengths",action='store_true')
    parser.add_argument("--blockdists", help="",action='store_true')
    parser.add_argument("--blockgaps", help="",action='store_true')
    parser.add_argument("--scrobblegaps", help="",action='store_true')
    parser.add_argument("--ee_artists",help="",action='store_true')
    parser.add_argument("--ee_artists_2",help="",action='store_true')
    parser.add_argument("--ee_artists_dists",help="",action='store_true')
    parser.add_argument("--block_len_dists",help="",action='store_true')



    args = parser.parse_args()

    if args.file is None:
        from pathos.multiprocessing import ProcessingPool as Pool

    mpa = setup(args,logging_level=logging.INFO)
    mpa.run()





#python setup.py -r -p -s -f /Users/jaredlorince/git/MusicForaging/testData/scrobbles/10000933.txt --suppdir /Users/jaredlorince/git/MusicForaging/GenreModeling/data/ --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ --feature_path /Users/jaredlorince/git/MusicForaging/GenreModeling/data/features/lda_artists/features_190.npy --session_thresho 0 --min_patch_length 5 --dist_thresh 0.2

#python setup.py -f /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/10000933.pkl --suppdir /Users/jaredlorince/git/MusicForaging/GenreModeling/data/ --feature_path /Users/jaredlorince/git/MusicForaging/GenreModeling/data/features/lda_artists/features_190.npy --blockdists --resultdir /Users/jaredlorince/git/MusicForaging/testData/

#python setup.py --patch_basis patch_idx_shuffle --suppdir /Users/jaredlorince/git/MusicForaging/GenreModeling/data/ --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ -n 4 --feature_path /Users/jaredlorince/git/MusicForaging/GenreModeling/data/features/lda_artists/features_190.npy

