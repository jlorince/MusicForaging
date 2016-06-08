#Wrapper sript for running analysis on BigRed2

rootdir='/N/u/jlorince/BigRed2/MusicForaging/foragingAnalysis/MPA/'


#### Just generate distances under a given feature space

#python ${rootdir}setup.py -f $1 -p --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy

#### Generate patches, blocks, dists, etc., but ignore session thresholding
#python ${rootdir}setup.py -f $1 -p --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --session_thresh 0 --dist_thresh 0.2 --min_patch_length 5

for dist_thresh in `seq 0.1 0.1 0.9`; do

    python setup.py -p -f $1 --patch_len_dist both --dist_thresh $dist_thresh --min_patch_length 2 --session_thresh 0 --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --resultdir /N/dc2/scratch/jlorince/patch_len_dists/

    for mpl in `seq 3 1 10`; do

        python setup.py -p -n 32 --patch_len_dist shuffle --dist_thresh $dist_thresh --min_patch_length $mpl --session_thresh 0 --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy --resultdir /N/dc2/scratch/jlorince/patch_len_dists/

    done

done
