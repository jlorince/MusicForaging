"""
Simple spark script to calculate all occuring sequential pairs of artists. That is, determine all unique pairs of artists A and B where
"""

### SPARK VERSION

# Here's a little function turn a sequence into tuples of sequential pairs
import itertools
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

# load the raw data
raw_data = sc.textFile("gs://music-foraging/lastfm_scrobbles.txt").repartition(128) # add appropriate repartitioning
#raw_data = sc.textFile("../testData/myListening.txt")

# parse the data, pulling out just user IDs and artist IDs
parsed = raw_data.map(lambda row: row.strip().split('\t')).map(lambda row: (int(row[0]),int(row[2]),row[-1]))

# group by user id, then sort by timestamp
grouped = parsed.groupBy(lambda row: row[0]).map(lambda row: [rw[1] for rw in sorted([i for i in row[1]],key=lambda val: val[-1])])

# generate all unique pairs for each user
paired = grouped.map(lambda row: set([tuple(sorted(pair)) for pair in pairwise(row) if pair[1]!=pair[0]]))

# reduce sets together
result = paired.reduce(lambda a,b: a.union(b))

### GRAPHLAB VERSION

import graphlab as gl

data = gl.SFrame.read_csv("testData/mylistening.txt",header=None,delimiter='\t')
data.rename({'X1':'user_id','X2':'item_id','X3':'artist_id','X4':'ts'})

grp = data.groupby(['user_id'],{'listening':gl.aggregate.CONCAT("ts","artist_id")})

q=grp.apply(lambda x: set(pairwise([val[1] for val in sorted(x.iteritems())])))


def srt(pair):
    if pair[0]>pair[1]:
        return (pair[1],pair[0])
    return pair

#filepath = 'scrobbles_sorted_user_ts.txt'
#filerows = 4691766834
filepath = 'testData/myListening.txt'
filerows = 58833

last_artist = None
last_user = None
comps = set()
with open(filepath) as fin:
    for i,line in enumerate(fin):
        if i%10000 == 0:
            print "%s/%s (%.02f%%)" % (i,filerows,100*(float(i)/filerows))
        line = line.strip().split('\t')
        user_id, artist_id = line[0],line[2]
        if user_id != last_user:
            last_artist = artist_id
            last_user = user_id
        else:
            if (last_artist!=None) and (last_artist!=artist_id):
                comps.add(srt((last_artist,artist_id)))
            last_artist = artist_id




