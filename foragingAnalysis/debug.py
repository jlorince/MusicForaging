import sys
import multiprocessing as mp
import glob
import pandas as pd


def get_count(f):
    return len(f.readlines())



if __name__ == '__main__':
    files = glob.glob('/Users/jaredlorince/git/MusicForaging/testData/*tiny.txt')
    print files
    pool = mp.Pool(32) # use 4 processes
    n =0
    for r in pool.imap_unordered(raw_processor, files):
        n+= r
    print n

