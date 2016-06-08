for dist_thresh in `seq 0.1 0.1 0.9`; do

    python setup.py -p -n 32 --patch_len_dist both --dist_thresh $dist_thresh --min_patch_length 2 --session_thresh 0 --feature_path /home/jlorince/lda_tests_artists/features_190.npy
    #python setup.py -p --suppdir /Users/jaredlorince/git/MusicForaging/GenreModeling/data/ -n 4 --patch_len_dist both --dist_thresh $dist_thresh --min_patch_length 2 --session_thresh 0 --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ --resultdir ./ --feature_path /Users/jaredlorince/git/MusicForaging/GenreModeling/data/features/lda_artists/features_190.npy

    for mpl in `seq 3 1 10`; do

        python setup.py -p -n 32 --patch_len_dist shuffle --dist_thresh $dist_thresh --min_patch_length $mpl --session_thresh 0 --feature_path /home/jlorince/lda_tests_artists/features_190.npy
        #python setup.py -p -n 4 --patch_len_dist shuffle --dist_thresh $dist_thresh --min_patch_length $mpl --session_thresh 0 --suppdir /Users/jaredlorince/git/MusicForaging/GenreModeling/data/ --pickledir /Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/ --resultdir ./ --feature_path /Users/jaredlorince/git/MusicForaging/GenreModeling/data/features/lda_artists/features_190.npy

    done

done
