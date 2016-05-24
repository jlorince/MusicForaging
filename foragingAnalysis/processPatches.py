import pandas as pd
from glob import glob
import multiprocessing as mp
import sys
import argparse
from scipy.spatial.distance import cosine,euclidean,pdist
import numpy as np
from itertools import chain,tee, izip
from functools import partial
import parmap
import time
import datetime
import logging

##### HELPER FUNCTIONS

# Jensen Shannon Distance (Sqrt of Jensen Shannon Divergence)
def JSD(P, Q):
    if np.all(np.isnan(P)) or np.all(np.isnan(Q)):
        return np.nan
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return np.sqrt(np.clip(0.5 * (entropy(_P, _M) + entropy(_Q, _M)),0,1))

# Calculate distance between any two feature arrays
def calc_dist(features1,features2,metric='cosine'):
    if np.any(np.isnan(features1)) or np.any(np.isnan(features2)):
        return np.nan
    if np.all(features1==features1):
        return 0.0
    if metric == 'JSD':
        return JSD(features1,features2)
    elif metric == 'cosine':
        return cosine(features1,features2)
    elif metric == 'euclidean':
        return euclidean(features1,features2)

# Get features for an artist, given an artist index in the feature matrix
def get_features(idx):
    if idx:
        return features[idx]
    else:
        return np.repeat(np.nan,features.shape[1])

# "s -> (s0,s1), (s1,s2), (s2, s3), ..."
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def patch_segmenter(df,metric,min_length,dist_thresh):
    l = df['artist_id']
    indices = list(np.array([len(list(v)) for g,v in itertools.groupby(l)][:-1]).cumsum())
    new_indices = []

    for b in indices:
        dist = calc_dist(df.iloc[b]['features'],df.iloc[b-1]['features'], metric=metric)

        if (np.isnan(dist)) or (dist >= dist_thresh):
            new_indices.append(b)

    if new_indices:

        last_patch = False
        final_indices = []
        for i,(a,b) in enumerate(pairwise([0]+new_indices+[len(df)])):
            if b-a>=min_length:
                final_indices.append(a)
                last_patch = True
            else:
                if last_patch:
                    final_indices.append(a)
                last_patch = False

        return final_indices,new_indices

    return new_indices,new_indices


def processor(fi,output_path,is_sorted=True,feature_path=None,dist='cosine',session_threshold=None,dist_threshold=0.2,min_patch_length=5,artist_idx_feature_map=None):

    # get user_id from filename
    user = fi.split('/')[-1][:-4]

    if f.endswith('.txt'):

        if output_path is None:
            raise("output path must be specified!")
        if artist_idx_feature_map_path is None:
            raise("artist_idx_feature_map_path must be provided!")


        # load table
        # df = pd.concat([pd.read_table(
        #         fi,
        #         header=None,
        #         names=['user_id','artist_id','ts'],
        #         parse_dates=['ts'],usecols=['user_id','artist_id','ts'])\
        #     .sort_values(by='ts') for f in glob(fi+'/part*')])
        df = pd.read_table(fi,
                header=None,
                names=['artist_id','ts'],
                parse_dates=['ts'])
        if not is_sorted:
            df = df.sort_values(by='ts')


        df['td'] = df['ts']-df.shift(1)['ts']
        df['td'] = df['td'].astype(int) / 10**9
        df['artist_idx'] = df['artist_id'].apply(lambda x: artist_idx_feature_map.get(x))
        df['before'] = df.shift(1)['artist_idx']

    elif f.endswith('.pkl'):
        df = pd.read_pickle(fi)

    # get features and calculate distances
    if feature_path is not None:
        features = np.load(feature_path)
        df['features'] = df['artist_idx'].apply(lambda idx: get_features(idx))
        df['features_shift'] = df['features'].shift(1)

        df['dist'] = df.apply(lambda row: calc_dist(row['features'],row['features_shift'],metric=dist),axis=1)

    if session_threshold is not None:
        session_idx = 0
        session_indices = []
        for val in df['td']>=session_threshold:
            if val:
                session_idx +=1
            session_indices.append(session_idx)
        df['session'] = session_indices

    if (min_patch_length is not None) and (dist_thresh is not None):

        indices_shuffle = np.zeros(len(df),dtype=int)
        indices_simple = np.zeros(len(df),dtype=int)
        offset_shuffle = 0
        idx_shuffle=0
        offset_simple = 0
        idx_simple=0

        ### NEED TO REWORK THIS BIT TO LOSE SOME REDUNDANCY

        for session in test.groupby('session'):
            result_shuffle,result_simple = patch_segmenter(session[1],metric=metric,min_length=min_patch_length,dist_thresh=dist_threshold)
            n=len(session[1])

            if len(result_shuffle)==0:
                indices_shuffle[offset_shuffle:offset_shuffle+n] = idx_shuffle
                idx_shuffle+=1
            else:
                indices_shuffle[offset_shuffle:offset_shuffle+result_shuffle[0]] = idx_shuffle
                idx_shuffle+=1
                for v, w in pairwise(result_shuffle):
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
                for v, w in pairwise(result_simple):
                    indices_simple[offset_simple+v:offset_simple+w] = idx_simple
                    idx_simple+=1
                indices_simple[offset_simple+result_simple[-1]:offset_simple+result_simple[-1]+n] = idx_simple
                idx_simple+=1
            offset_simple += n


        if result_shuffle:
            indices_shuffle[offset_shuffle+result_shuffle[-1]:] = idx
        else:
            indices_shuffle[offset_shuffle:] = idx

        if result_simple:
            indices_simple[offset_simple+result_simple[-1]:] = idx
        else:
            indices_simple[offset_simple:] = idx

        df['patch_idx_shuffle'] = indices_shuffle
        df['patch_idx_simple'] = indices_simple

        # add artist block info
        ### https://stackoverflow.com/questions/14358567/finding-consecutive-segments-in-a-pandas-data-frame

        df['block'] = (df['artist_id'].shift(1) != df['artist_id']).astype(int).cumsum()

    # save to pickle file and we're done
    df.to_pickle('{}{}.pkl'.format(output_dir,user))
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",action="store_true")
    parser.add_argument("-p", "--preprocess", help="perform preprocessing of listening histories",action="store_true")
    parser.add_argument("--rawtext", help="Load scrobbles from raw text files. If not specififed, assumes files are already pickled and saved in `pickledir`",action="store_true")
    parser.add_argument("--pickledir", help="specify output dir for pickled dataframes",action="store_true",default='/home/jlorince/scrobbles_processed/')
    parser.add_argument("-d","--datadir", help="specify base directory containing input files",action="store_true",default='/home/jlorince/scrobbles/')
    parser.add_argument("-s","--suppdir", help="specify supplementary data location",action="store_true",default='/home/jlorince/support/')
    parser.add_argument("-r","--resultdir", help="specify results location",action="store_true",default='/home/jlorince/results/')
    parser.add_argument("-t","--threshold", help="session segmentation threshold",type=int,default=1800)
    parser.add_argument("-m","--min_patch_length", help="minimum patch length",type=int,default=5)
    parser.add_argument("--dist_thresh", help="distance threshold defining patch neigborhood",type=float,default=0.2)
    parser.add_argument("-n","--num_processes", help="number of processes in processor pool",type=int,default=2)
    parser.add_argument("-f","--feature_path", help="path to artist feature matrix",action="store_true",default='/home/jlorince/lda_tests_artists/features_190.npy')
    parser.add_argument("--distance_metric", help="distance metric",type=str,default='cosine')



    args = parser.parse_args()
    logging.info(str(args))


    #files = glob('/Users/jaredlorince/git/MusicForaging/testData/scrobbles/*.txt')

    if args.preprocess:

        logging.info("")

        start = time.time()
        artist_idx_feature_map = {}
        for line in open('artist_idx_feature_map_path'):
            k,v = line.strip().split('\t')
            artist_idx_feature_map[float(k)] = int(v)

        pool = mp.Pool(args.num_proccesses)
        if args.rawtext:
            files = glob(arg.datadir)
        else:
            files = glob(arg.pickledir)
        func_partial = partial(processor,output_path=args.pickledir,is_sorted=True,feature_path=args.feature_path,dist=arg.distance_metric,session_threshold=args.dist_thresh,min_patch_length=args.min_patch_length,artist_idx_feature_map=artist_idx_feature_map)
        pool.map(func_partial,files)
        pool.close()


    #processor(fi,output_path,is_sorted=True,feature_path=None,dist='cosine',session_threshold=None,dist_threshold=0.2,min_patch_length=5)


 #   python processPatches.py -p --rawtext -d '/Users/jaredlorince/git/MusicForaging/testData/scrobbles/' --pickledir '/Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/' -s '/Users/jaredlorince/git/MusicForaging/testData/support/'

