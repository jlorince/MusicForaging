#Wrapper sript for running analysis on BigRed2

rootdir='/N/u/jlorince/BigRed2/MusicForaging/foragingAnalysis/MPA/'


#### Just generate distances under a given feature space

#python ${rootdir}setup.py -f $1 -p -s  --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy

#### Generate patches, blocks, dists, etc., but ignore session thresholding

#python ${rootdir}setup.py -f $1 -p --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed_2_10/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --session_thresh 0 --dist_thresh 0.2 --min_patch_length 10

#### param sensitibity analyis

# for dist_thresh in `seq 0.1 0.1 0.9`; do

#     #python ${rootdir}setup.py -p -f $1 --patch_len_dist both --dist_thresh $dist_thresh --min_patch_length 2 --session_thresh 0 --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --resultdir /N/dc2/scratch/jlorince/patch_len_dists/ --suppdir /N/dc2/scratch/jlorince/support/

#     for mpl in `seq 2 1 10`; do

#         python ${rootdir}setup.py -p -f $1 --patch_len_dist --dist_thresh $dist_thresh --min_patch_length $mpl --session_thresh 0 --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --resultdir /N/dc2/scratch/jlorince/patch_len_dists/ --suppdir /N/dc2/scratch/jlorince/support/

#     done

# done

#### JUMP DISTRIBUTIONS

#python ${rootdir}patchAnalyses.py -f $1 --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --resultdir /N/dc2/scratch/jlorince/jumpdists/ --jumpdists

#### PATCH STATS
#python ${rootdir}setup.py -f $1 --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed_2_5/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --patch_basis patch_idx_shuffle --resultdir /N/dc2/scratch/jlorince/patches_shuffle_2_5/

#### BLOCK DISTS
#python ${rootdir}setup.py -f $1 --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed_2_10/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --blockdists --resultdir /N/dc2/scratch/jlorince/blockdists_random/

#### PATCH BLOCK DISTS
#python ${rootdir}patchAnalyses.py -f $1 --pickledir /N/dc2/scratch/jlorince/patches/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --resultdir /N/dc2/scratch/jlorince/patch_blockdists/ --blockdists

#### PATCH DIVERSITY DISTS
python ${rootdir}patchAnalyses.py -f $1 --pickledir /N/dc2/scratch/jlorince/patches/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --resultdir /N/dc2/scratch/jlorince/patch_diversity_dists/ --diversity_dists
