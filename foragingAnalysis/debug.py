import sys
#import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
import glob
import pandas as pd
from functools import partial
import time
import logging
import time
import datetime


def get_count(f):
    return len(f.readlines())

class C(object):
    def __init__(self,files):
        self.pool = Pool(4)
        self.files = files

    def raw_processor(self, fi,prefix,somedict):
        df = pd.read_table(
                fi,
                header=None,
                names=['artist_id','ts'],
                parse_dates=['ts'])\
            .sort_values(by='ts')
        user = fi.split('/')[-1][:-4]
        df.to_pickle('/Users/jaredlorince/git/MusicForaging/testData/scrobbles_test/{}_{}.pkl'.format(prefix,user))
        rootLogger.info('preprocessing complete for user {} ({})'.format(user,fi))

    def run_p(self):
        func_partial = partial(self.raw_processor,prefix='blah',somedict=d)
        result = self.pool.map(func_partial, self.files)




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

    # pool = mp.Pool(4) # use 4 processes
    # func_partial =partial(raw_processor,prefix='blah',somedict=d)
    # result = pool.map(func_partial, files)
    # pool.close()

    # pool.close()
    # pool.join()
    c = C(files)
    c.run_p()

    #pool.map_async(raw_processor,)


