import pandas as pd
from glob import glob
#from pathos.multiprocessing import ProcessingPool as Pool
import sys
import argparse
from scipy.spatial.distance import cosine,euclidean,pdist
import numpy as np
from itertools import chain,tee, izip, groupby
from functools import partial
import time
import datetime
import logging
import warnings
import setup
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy import sparse


class analyze(setup.setup):

    def __init__(self,args,logging_level=logging.INFO):

         super(analyze, self ).__init__(args,logging_level)


    # set up processing pool and run all analyses specified in args
    def run(self):


        if self.args.jumpdists:
            n_bins=100.
            bin_width = 1/n_bins
            bins = np.arange(0,1+bin_width,1/n_bins)

            if self.args.file:
                user,vals = self.artist_jump_distributions(self.args.file,bins=bins,self_jumps=False)
                with open(self.args.resultdir+user,'w') as fout:
                    fout.write(','.join(vals.astype(str))+'\n')



            else:
                raise('not implemented!')
                self.pool = Pool(self.args.n)
                self.rootLogger.info("Pool started")

                self.rootLogger.info("Starting jump distance analysis")

                func_partial = partial(self.artist_jump_distributions,bins=bins,self_jumps=False)
                with open(self.args.resultdir+'jumpdists','w') as fout:
                    for user,vals in self.pool.imap(func_partial,self.listen_files):
                        fout.write(user+'\t'+','.join(vals.astype(str))+'\n')

                self.pool.close()
                self.rootLogger.info("Pool closed")

        if self.args.blockdists:
            #self.rootLogger.info("Starting block distance analysis")
            self.mean_block_distances(self.args.file)

        if self.args.diversity_dists:
            bins = np.arange(0,1.01,.01)
            self.diversity_distributions(self.args.file,bins=bins)

        if self.args.clustering:
            self.clustering(self.args.file)

        if self.args.values:
            self.patch_values(self.args.file)

        if self.args.exp:
            self.explore_exploit(self.args.file)

        if self.args.patch_len_dists:
            self.patch_len_dists(self.args.file)


    # calculate distribution (using histogram with specified bins)
    # of sequential artist-to-artist distances
    def artist_jump_distributions(self,fi,bins,self_jumps=False):
        user = fi.split('/')[-1][:-4]
        df = pd.read_pickle(fi)
        if self_jumps:
            vals = np.histogram(df['dist'].dropna(),bins=bins)[0]
        else:
            vals = np.histogram(df['dist'][df['dist']>0],bins=bins)[0]
        self.rootLogger.info('artist jump distances done for user {} ({})'.format(user,fi))
        return user,vals

    # calculate distribution (using histogram with specified bins)
    # of patch diversity for each user

    # awk 'FNR==1' * > diversity_dists_zeros
    # awk 'FNR==2' * > diversity_dists_nozeros
    def diversity_distributions(self,fi,bins):
        if 'patches' not in fi:
            raise('WRONG DATATYPE')
        user = fi.split('/')[-1].split('_')[0]
        df = pd.read_pickle(fi).dropna(subset=['diversity'])
        zeros = np.histogram(df[df['n']>=5]['diversity'],bins=bins)[0]
        nozeros = np.histogram(df[(df['n']>=5)&(df['diversity']>0)]['diversity'],bins=bins)[0]

        zeros = zeros/float(zeros.sum())
        nozeros = nozeros/float(nozeros.sum())

        with open(self.args.resultdir+user,'w') as fout:
            fout.write(user+'\t'+'zeros'+'\t'+','.join(zeros.astype(str))+'\n')
            fout.write(user+'\t'+'nozeros'+'\t'+','.join(nozeros.astype(str))+'\n')
        self.rootLogger.info('diversity distributions done for user {} ({})'.format(user,fi))


    def mean_block_distances(self,fi,n=100):

        def cos_nan(arr1,arr2):
            if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
                return np.nan
            else:
                return cosine(arr1,arr2)


        user = fi.split('/')[-1].split('_')[0]
        df = pd.read_pickle(fi)
        blocks = df[df['n']>=5].dropna()

        result = []
        for i in xrange(len(blocks)-n):
            first = blocks['centroid'].iloc[i]
            result.append(np.array(blocks['centroid'][i+1:i+n+1].apply(lambda val: cos_nan(val,first))))
        result = np.nanmean(np.vstack(result),0)

        with open(self.args.resultdir+user,'w') as fout:
            fout.write('\t'.join([user,'patch',','.join(result.astype(str))])+'\n')

        self.rootLogger.info('Block distances for user {} processed successfully ({})'.format(user,fi))


        # now shuffled
        # idx = np.array(blocks.index)
        # np.random.shuffle(idx)
        # blocks = blocks.reindex(idx)

        # result_random = []
        # for i in xrange(len(blocks)-n):
        #     first = blocks['centroid'].iloc[i]
        #     result_random.append(np.array(blocks['centroid'][i+1:i+n+1].apply(lambda val: cos_nan(val,first))))
        # result_random = np.nanmean(np.vstack(result_random),0)

        # with open(self.args.resultdir+user,'w') as fout:
        #     fout.write('\t'.join([user,'patch',','.join(result.astype(str))])+'\n')
        #     fout.write('\t'.join([user,'patch_random',','.join(result_random.astype(str))])+'\n')
        # self.rootLogger.info('Block distances for user {} processed successfully ({})'.format(user,fi))

    def clustering(self,fi):
        df = pd.read_pickle(fi)
        user = fi.split('/')[-1].split('_')[0]

        mask = (df['centroid'].apply(lambda arr: ~np.any(np.isnan(arr))).values)&(df['n']>=5)&(df['diversity']<=0.2)
        clust_data = df[mask].reset_index()
        arr =  np.vstack(clust_data['centroid'])
        Z = linkage(arr, 'complete')
        clusters = fcluster(Z,t=0.2,criterion='distance')
        assignments = np.repeat(np.nan,len(df))
        assignments[np.where(mask)] = clusters
        df['patch_clust'] = assignments
        df.to_pickle('{}{}.pkl'.format(self.args.resultdir,user))
        self.rootLogger.info('Patch clusters for user {} processed successfully ({})'.format(user,fi))

    def patch_len_dists(self,fi):
        df = pd.read_pickle(fi)
        user = fi.split('/')[-1][:-4]

        explore = df[np.isnan(df['patch_clust'])]
        exploit = df[~np.isnan(df['patch_clust'])]

        result_explore = explore['n'].value_counts()
        result_exploit = exploit['n'].value_counts()

        result_explore = result_explore.reindex(xrange(1,max(result_explore.index)+1),fill_value=0.).values
        result_exploit = result_exploit.reindex(xrange(1,max(result_exploit.index)+1),fill_value=0.).values

        result_explore = sparse.csr_matrix(result_explore)
        result_exploit = sparse.csr_matrix(result_exploit)


        with open(self.args.resultdir+user,'w') as fout:
            fout.write(user+'\t'+'explore'+'\t'+':'.join([','.join(a.astype(str)) for a in result_explore.data,result_explore.indices,result_explore.indptr])+'\n')
            fout.write(user+'\t'+'exploit'+'\t'+':'.join([','.join(a.astype(str)) for a in result_exploit.data,result_exploit.indices,result_exploit.indptr])+'\n')
        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))






    def explore_exploit(self,fi):

        user = fi.split('/')[-1][:-4]

        df_patches_raw = pd.read_pickle(fi)

        # add time in next bout
        df_patches_raw['next_n'] = df_patches_raw['n'].shift(-1)

        # add patch values
        # listensPerPatch = df_patches_raw.groupby('patch_clust')['n'].sum()
        # overall_prop = listensPerPatch/float(df_patches_raw['n'].sum())
        # overall_prop.name = 'final_value'
        # df_patches_raw = df_patches_raw.join(overall_prop,on='patch_clust')


        """
        # time in next exploit patch as function of exploration time
        result = df_patches_raw[np.isnan(df_patches_raw['patch_clust'])].groupby('n')['next_n'].mean()

        fout.write(user+'\t'+'next-exploit-vs-explore'+'\t'+','.join(["{}:{}".format(a,b) for a,b in result.iteritems()])+'\n')
        """
        # total time exploiting as a function of time exploring
        df_patches_raw['explore'] = np.isnan(df_patches_raw['patch_clust']).astype(int)
        df_patches_raw['explore-idx'] = df_patches_raw['explore'].cumsum()
        grp_explore = df_patches_raw.groupby('explore-idx').apply(lambda df: pd.DataFrame({'n':[df['n'].iloc[0]],'n-exploit':[df['n'][1:].sum()]}))

        #result = grp_explore.groupby('n')['n-exploit'].mean()
        #fout.write(user+'\t'+'total-exploit-vs-explore'+'\t'+','.join(["{}:{}".format(a,b) for a,b in result.iteritems()])+'\n')
        """
        # exploration time as a function of exploitation time
        grp_exploit = grp_explore.copy()
        grp_exploit['n-explore'] = grp_exploit['n'].shift(-1)

        result = grp_exploit.groupby('n-exploit')['n-explore'].mean()
        fout.write(user+'\t'+'explore-vs-exploit'+'\t'+','.join(["{}:{}".format(a,b) for a,b in result.iteritems()])+'\n')
        """
        # prob exploit given explore time
        explore_only = df_patches_raw[np.isnan(df_patches_raw['patch_clust'])]
        result = explore_only['n'][:-1].value_counts()
        arr = result.reindex(xrange(1,max(result.index)+1),fill_value=0.).values
        final_result = arr/(np.cumsum(arr[::-1])[::-1])
        final_result = sparse.csr_matrix(final_result)

        with open(self.args.resultdir+user+'_exploit','w') as fout:
            fout.write(user+'\t'+':'.join([','.join(a.astype(str)) for a in final_result.data,final_result.indices,final_result.indptr])+'\n')

        #fout.write(user+'\t'+'prob-exploit-given-explore'+'\t'+','.join(["{}:{}".format(a,b) for a,b in result.iteritems()])+'\n')

        # prob explore given exploit time
        result = grp_explore['n-exploit'][grp_explore['n-exploit']>0].value_counts()
        arr = result.reindex(xrange(1,max(result.index)+1),fill_value=0.).values
        final_result = arr/np.cumsum(arr[::-1])[::-1]
        final_result = sparse.csr_matrix(final_result)

        with open(self.args.resultdir+user+'_explore','w') as fout:
            fout.write(user+'\t'+':'.join([','.join(a.astype(str)) for a in final_result.data,final_result.indices,final_result.indptr])+'\n')


        #fout.write(user+'\t'+'prob-explore-given-exploit'+'\t'+','.join(["{}:{}".format(a,b) for a,b in result.iteritems()])+'\n')

        """
        # patch value as a function of exploration time
        df_patches_raw['final_value_next'] = df_patches_raw['final_value'].shift(-1)
        result = df_patches_raw[np.isnan(df_patches_raw['patch_clust'])].groupby('n')['final_value_next'].mean()
        fout.write(user+'\t'+'exploit-value-vs-explore'+'\t'+','.join(["{}:{}".format(a,b) for a,b in result.iteritems()])+'\n')
        """

        self.rootLogger.info('User {} processed successfully ({})'.format(user,fi))













if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--resultdir", help="specify results location",default='/home/jlorince/results/')
    parser.add_argument("--pickledir", help="specify output dir for pickled dataframes",default='/home/jlorince/scrobbles_processed/')
    parser.add_argument("-n", help="number of processes in processor pool",type=int,default=1)
    parser.add_argument("--feature_path", help="path to artist feature matrix",default=None) # '/home/jlorince/lda_tests_artists/features_190.npy'
    parser.add_argument("-f", "--file",help="If provided, run analysis for this file only",default=None,type=str)

    ### These are all the analyses we can run:
    parser.add_argument("--jumpdists", help="generate each user's distribution of artist-artist distances",action='store_true')
    parser.add_argument("--blockdists", help="",action='store_true')
    parser.add_argument("--diversity_dists", help="generate distribution of patch diversity for each user",action='store_true')
    parser.add_argument("--clustering", help="apply patch clustering",action='store_true')
    parser.add_argument("--values", help="apply patch clustering",action='store_true')
    parser.add_argument("--exp", help="explore/exploit analyses",action='store_true')
    parser.add_argument("--patch_len_dists", help="explore/exploit segment length dists",action='store_true')





    args = parser.parse_args()

    if args.file is None:
        from pathos.multiprocessing import ProcessingPool as Pool


    mpa = analyze(args,logging_level=logging.INFO)
    mpa.run()

    #python patchAnalyses.py --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ -n 4 --resultdir ./ --jumpdists

    #python patchAnalyses.py --feature_path /Users/jaredlorince/git/MusicForaging/GenreModeling/data/features/lda_artists/features_190.npy --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ -n 4 --resultdir ./ --blockdists
