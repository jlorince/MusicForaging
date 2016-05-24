import sys
import multiprocessing as mp
import glob
import time


def get_count(f):
    return len(open(f).readlines())



if __name__ == '__main__':
    start = time.time()
    files = glob.glob('scrobbles/*txt')
    pool = mp.Pool(24) # use 4 processes
    n =0
    for r in pool.imap_unordered(get_count, files):
        n+= r
    print n,(time.time()-start)/60.


#gcloud compute copy-files --zone "us-east1-c" scrobbles "jlorince@ssd:~/"


