"""
Assumed to be running on Karst/RDC at IU.

Generates two files, one consisting of the sequence of artist IDs listened to by each user, the other consisting of the time deltas (i.e. time gap between listens) for all scrobbles by that user.
"""

import MySQLdb
import time
import datetime
import sys
import os
import cPickle
import itertools

wall_time = 24 * 60 * 60 # 24hr
time_buffer = 15 * 60 # 15min

start = time.time()

### suprress warnings
from warnings import filterwarnings
filterwarnings('ignore', category = MySQLdb.Warning)

#db = MySQLdb.connect(host="127.0.0.1", user="root", passwd="lasagna",db="analysis_lastfm")
db=MySQLdb.connect(host='rdc04.uits.iu.edu',port=3094,user='root',passwd='jao78gh',db='analysis_lastfm')

cursor=db.cursor()
cursor.execute("SET SQL_BIG_SELECTS=1;")
cursor.execute("SET time_zone = '+00:00';")
cursor.execute("set sql_select_limit=18446744073709551615;")

cursor.execute("select user_id from lastfm_users where scrobbles_recorded=1")
users = [u[0] for u in cursor.fetchall()]

seq_file = '/N/dc2/scratch/jlorince/artist_seqs'
delta_file = '/N/dc2/scratch/jlorince/time_deltas'

done = set()
try:
    with open(seq_file) as fin:
        for line in fin:
            done.add(int(line[:line.find('\t')]))
except IOError:
    pass


with open(seq_file,'a') as seqs, open(delta_file,'a') as deltas:
    for u in users:
        if time.time()-start >= (wall_time - time_buffer):
            cursor.close()
            db.close()
            sys.exit()

        if u in done:
            continue

        cursor.execute("select artist_id,scrobble_time from lastfm_scrobbles where user_id=%s order by scrobble_time asc" % u)
        result = cursor.fetchall()

        seq = [sc[0] for sc in result]

        delta_vals = []
        for i,val in enumerate(result[1:]):
            delta_vals.append(int((val[1]-result[i][1]).total_seconds()))

        seqs.write(str(u)+'\t'+' '.join(map(str,seq))+'\n')
        deltas.write(str(u)+'\t'+' '.join(map(str,delta_vals))+'\n')
        seqs.flush()
        deltas.flush()


