#Wrapper sript for running analysis on BigRed2

rootdir='/N/u/jlorince/BigRed2/MusicForaging/foragingAnalysis/MPA/'


#### Just generate distances under a given feature space

#python ${rootdir}setup.py -f $1 -p --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy

#### Generate patches, blocks, dists, etc., but ignore session thresholding
python ${rootdir}setup.py -f $1 -p --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --session_thresh 0 --dist_thresh 0.2 --min_patch_length 5
