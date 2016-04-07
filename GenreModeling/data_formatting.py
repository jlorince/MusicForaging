#### Convert spark-generated LDA vectors to Graphlab SArray

print "loading raw data..."
raw_docs = gl.SArray("LDA_vectors/")

print "loading vocab indices"
vocab_idx = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    vocab_idx[int(line[1])] = line[0]
    #vocab_idx[line[0]] = int(line[1])

print "loading bad_data"
bad = set([int(aid.strip()) for aid in open('bad_artists').readlines()])

def formatter(row):
    row = eval(row)
    result = {}
    for term,cnt in row[1]:
        artist_name = vocab_idx[term]
        if artist_name not in bad:
            result[artist_name] = cnt
    #result = dict(zip([vocab_idx[term] for term in row[1][1]],row[1][2]))
    return (row[0],result)

print "formatting data..."
docs = raw_docs.filter(lambda x: x!="").apply(formatter)
print "filtering data..."
def filter_counts(x):
    xsum = sum(x[1].values())
    return (xsum>=1000) and (xsum<=500000)
#docs = docs.filter(lambda x: sum(x[1].values())>=1000)
docs = docs.filter(filter_counts)
user_idx = {row[0]:i for i,row in enumerate(docs)}
with open('user_idx','w') as fout:
    for k,v in user_idx.iteritems():
        fout.write(str(k)+'\t'+str(v)+'\n')
print "saving data..."
docs.save("doc_array",format='binary')
