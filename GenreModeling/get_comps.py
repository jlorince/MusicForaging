import time
import datetime
import sys
import os
import cPickle
import itertools

#wall_time = 24 * 60 * 60 # 24hr
#time_buffer = 15 * 60 # 15min

start = time.time()

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def srt(pair):
    if pair[0]>pair[1]:
        return (pair[1],pair[0])
    return pair


done_filepath = 'comps_processed'

try:
    done = set()
    with open(done_filepath) as fin:
        for line in fin:
            done.add(int(line.strip().split()[0]))
except IOError:
    pass


try:
    overall_comps = cPickle.load(open('comps.pkl','r'))
except IOError as e:
    overall_comps = set()
    #raise e

done_file = open(done_filepath,'a')


for line in open('artist_seqs'):
    #st = time.time()
    line = line.strip().split('\t')
    u = int(line[0])
    if (len(line)==1) or (u in done):
        #print "no scrobbles for user %s" % u
        continue
    result = map(int,line[1].split())

    #print 'read time: %s' % (time.time()-st)
    """
    st = time.time()
    cursor.execute("select artist_id from lastfm_scrobbles where user_id=%s order by scrobble_time asc" % u)
    result = [s[0] for s in cursor.fetchall()]
    print 'query time: %s' % (time.time()-st)
    """
    #st = time.time()
    comps = set([srt(i) for i in pairwise(result) if i[0]!=i[1]])
    #print 'comp time: %s' % (time.time()-st)
    #st = time.time()
    for comp in comps:
        overall_comps.add(comp)
    #       overall_comps = overall_comps.union(comps)
   # print 'union time: %s' % (time.time()-st)
    #print '-------------------------'
        te(str(u)+'\t' + str(len(overall_comps)) +'\n')
    done_file.flush()

cPickle.dump(overall_comps,open('comps.pkl','w'))
with open('all_comps','w') as fout:
    for comp in overall_comps:
        fout.write(str(comp[0])+'\t'+str(comp[1])+'\n')

sys.exit()



