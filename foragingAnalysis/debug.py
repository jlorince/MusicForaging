import sys
import multiprocessing as mp
import glob
import pandas as pd






if __name__ == '__main__':
    files = glob.glob('/Users/jaredlorince/git/MusicForaging/testData/*tiny.txt')
    print files
    pool = mp.Pool(2) # use 4 processes
    for r in pool.imap_unordered(raw_processor, files):
        print r
