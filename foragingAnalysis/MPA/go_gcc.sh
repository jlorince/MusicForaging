for dist_thresh in `seq 0.05 0.05 0.9`; do

    python setup.py -p --suppdir /Users/jaredlorince/git/MusicForaging/GenreModeling/data/ -n 32 --feature_path /home/jlorince/lda_tests_artists/features_190.npy --patch_len_dist both --dist_thresh $dist_thresh --min_patch_length 2

    for mpl in `seq 3 1 10`; do

        python setup.py -p --suppdir /Users/jaredlorince/git/MusicForaging/GenreModeling/data/ -n 32 --feature_path /home/jlorince/lda_tests_artists/features_190.npy --patch_len_dist both --dist_thresh $dist_thresh --min_patch_length $mpl

    done

done
