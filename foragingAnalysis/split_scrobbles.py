# awk approach

# awk '{print >> "scrobbles/"$1".txt"; close("scrobbles/"$1".txt")}' lastfm_scrobbles.txt

# Spark approach
# IPYTHON=1 spark-1.6.0-bin-hadoop2.6/bin/pyspark --driver-memory=190G

# raw_data = sc.textFile('lastfm_scrobbles.txt').map(lambda row: row.split('\t')).map(lambda row: (row[0],'\t'.join([row[2],row[3]])))
# print 'raw data loaded'

# df = raw_data.toDF(['userid','data'])
# print 'DF conversion complete'

# df.write.partitionBy("userid").text("scrobbles2")

# qsub -I -q interactive -l nodes=1:ppn=1,walltime=2:00:00
# qsub -l nodes=1:ppn=1,vmem=16gb,walltime=24:00:00 -m e go.sh

import MySQLdb
import sys
import logging
import datetime

now = datetime.datetime.now()
log_filename = now.strftime('splitter_%Y%m%d_%H%M%S.log')
logFormatter = logging.Formatter("%(asctime)s\t[%(levelname)s]\t%(message)s")
rootLogger = logging.getLogger()
fileHandler = logging.FileHandler(log_filename)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)


done = set()
if len(sys.argv)>1:
    logfiles = sys.argv[1:]
    for fi in logfiles:
        with open(fi) as f:
            for line in f:
                line = line.strip().split('\t')
                done.add(int(line[-1].split()[0]))

db=MySQLdb.connect(host='rdc04.uits.iu.edu',port=3094,user='root',passwd='jao78gh',db='analysis_lastfm')

cursor=db.cursor()
cursor.execute("SET SQL_BIG_SELECTS=1;")
cursor.execute("SET time_zone = '+00:00';")
cursor.execute("set sql_select_limit=18446744073709551615;")

n_users = cursor.execute("select user_id from lastfm_users where sample_playcount>=0;")
users = [u[0] for u in cursor.fetchall()]

for i,u in enumerate(users):
    if u in done:
        continue
    rootLogger.info("{} ({}/{})".format(u,i+1,n_users))
    with open("/N/dc2/scratch/jlorince/scrobbles-complete/{}.txt".format(u),'w') as fout:
        n_scrobbles = cursor.execute("select item_id,artist_id,scrobble_time from lastfm_scrobbles where user_id={} order by scrobble_time asc;".format(u))
        for scrobble in cursor:
            fout.write('\t'.join(map(str,scrobble))+'\n')


