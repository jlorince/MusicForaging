import sys
import multiprocessing as mp
import glob
import pandas as pd
from functools import partial
import time
import logging
import time
import datetime


def get_count(f):
    return len(f.readlines())



def raw_processor(fi,prefix,somedict):
    df = pd.read_table(
            fi,
            header=None,
            names=['user_id','item_id','artist_id','ts'],
            parse_dates=['ts'],usecols=['artist_id','ts'])\
        .sort_values(by='ts')
    user = fi.split('/')[-1][:-4]
    df.to_pickle('/Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/{}_{}.pkl'.format(prefix,user))
    rootLogger.info('preprocessing complete for {}'.format(fi))




if __name__ == '__main__':

    now = datetime.datetime.now()
    log_filename = now.strftime('%Y%m%d_%H%M%S.log')
    # logging.basicConfig(filename=log_filename,level=logging.DEBUG,filemode='a',format='%(asctime)s\t%(levelname)s:%(message)s:%(threadName)s')

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)



    start = time.time()

    d = {'a':1,'b':2}

    files = glob.glob('/Users/jaredlorince/git/MusicForaging/testData/scrobbles/*.txt')
    #files = glob.glob('/Users/jaredlorince/git/MusicForaging/testData/scrobbles/*.txt')[:10]

    pool = mp.Pool(4) # use 4 processes
    func_partial =partial(raw_processor,prefix='blah',somedict=d)
    result = pool.map(func_partial, files)
    pool.close()

    # pool.close()
    # pool.join()

    #pool.map_async(raw_processor,)


