#Wrapper sript for running analysis on BigRed2

rootdir='/N/u/jlorince/BigRed2/MusicForaging/foragingAnalysis/MPA/'


#### Just generate distances under a given feature space

python ${rootdir}setup.py -f $1 -p --suppdir /N/dc2/scratch/jlorince/support/ --pickledir /N/dc2/scratch/jlorince/scrobbles_processed/ --feature_path /N/dc2/scratch/jlorince/support/features_190.npy


