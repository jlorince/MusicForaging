import pandas as pd
import glob
import multiprocessing as mp
import sys
import argparse


def get_features(idx):
    if idx:
        return features[idx]
    else:
        return np.repeat(np.nan,features.shape[1])

def raw_processor(f):
    df = pd.read_table(
            fi,
            header=None,
            names=['user_id','item_id','artist_id','ts'],
            parse_dates=['ts']).sort_values(by='ts')




if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity",action="store_true")
    parser.add_argument("--rawtext", help="load scrobbles from raw text files",action="store_true")
    parser.add_argument("--pickledir", help="lspecify output dir for pickled dataframes",action="store_true")
    parser.add_argument("-d","--datadir", help="specify base directory containing input files",action="store_true",default='/home/jlorince/scrobbles/')
    parser.add_argument("-s","--suppdir", help="specify supplementary data location",action="store_true",default='/home/jlorince/support/')
    parser.add_argument("-r","--resultdir", help="specify results location",action="store_true",default='/home/jlorince/results/')
    parser.add_argument("-t","--threshold", help="session segmentation threshold",type=int,default=1800)
    parser.add_argument("-m","--min_patch_length", help="minimum patch length",type=int,default=5)

    args = parser.parse_args()

    ### LOAD COMMAND LINE ARGUMENTS

    print args


    # files = glob.glob(basedir)
    # pool = mp.Pool(32) # use 4 processes
    # for r in pool.map_asyn(process_user, files):
    #     print r

