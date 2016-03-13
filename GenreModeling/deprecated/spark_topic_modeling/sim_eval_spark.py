import numpy as np
import os
from scipy import sparse
import graphlab as gl
from scipy.spatial.distance import cosine,euclidean
import time
import datetime

gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)


LDA_vectors_path = "LDA_vectors"

random_sample_size=10000

np.random.seed(99)
comps = set()
while len(comps)<random_sample_size:
    a = np.random.randint(0,112312)
    b= np.random.randint(0,112312)
    if a!=b:
        comp = tuple(sorted([a,b]))
        comps.add(comp)
comps = sc.parallelize(comps).sortBy(lambda x: x)

artist_dict = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    artist_dict[line[0]] = int(line[1])

def parse_parts(d):
    output = []
    files = os.listdir(d)
    for fi in files:
        output+=[eval(line)[1] for line in open(d+'/'+fi).readlines()]
    return np.array(output)

def parse_sparse_array(arr_string):
    evaluated = eval(arr_string)
    #user_id = evaluated[0]
    data = evaluated[1][2]
    indices = evaluated[1][1]
    length = evaluated[1][0]
    sparse_matrix = sparse.csr_matrix((data,([0]*len(indices),indices)),shape=(1,length))
    return sparse_matrix.todense().A.flatten()

def get_token_counts(arr_string):
    evaluated = eval(arr_string)
    data = evaluated[1][2]
    return sum(data)

#LDA_vectors = np.array(gl.SArray(LDA_vectors_path).filter(lambda x: x!='').apply(parse_sparse_array))
#token_counts = LDA_vectors.sum(1)[:,np.newaxis]
#token_counts = np.array(gl.SArray(LDA_vectors_path).filter(lambda x: x!='').apply(get_token_counts))[:,np.newaxis]
token_counts = np.array(sc.textFile(LDA_vectors_path).map(get_token_counts).collect())[:,np.newaxis]


#artist_topic_path = "scala_lda_tf/artist_topic_10"
#user_topic_path = "scala_lda_tf/user_topic_10"
for k in np.arange(85,161,5):
    artist_topic_path = "scala_lda_tfidf_50iter/artist_topic_"+str(k)
    user_topic_path = "scala_lda_tfidf_50iter/user_topic_"+str(k)

    artist_topic = np.loadtxt(artist_topic_path)
    artist_topic /= artist_topic.sum(0)

    user_topic = parse_parts(user_topic_path)

    topic_counts = (user_topic * token_counts).sum(0)
    artist_topic_freq = artist_topic*topic_counts
    artist_topic_probs = artist_topic_freq/artist_topic_freq.sum(1,keepdims=True)
    atp_bc = sc.broadcast(artist_topic_probs)


    e_dists = []
    c_dists = []
    for a,b in comps:
        e_dists.append(euclidean(artist_topic_probs[a],artist_topic_probs[b]))
        c_dists.append(cosine(artist_topic_probs[a],artist_topic_probs[b]))

    c_dists = comps.map(lambda x: cosine(atp_bc.value[x[0]],atp_bc.value[x[1]]))
    e_dists = comps.map(lambda x: euclidean(atp_bc.value[x[0]],atp_bc.value[x[1]]))


    e_dists = np.array(e_dists)
    c_dists = np.array(c_dists)
    print "euclidean (k=%s): mean=%s, var=%s" % (k,e_dists.mean(),e_dists.var())
    print "cosine (k=%s): mean=%s, var=%s" % (k,c_dists.mean(),c_dists.var())
    np.savetxt('e_'+str(k), e_dists)
    np.savetxt('c_'+str(k), c_dists)


def rmse(a,b):
    return np.sqrt(((a-b)**2).mean())
dist_dict = {'c_':'cosine','e_':'euclidean'}

for dist in dist_dict:
    rmse_vals = []
    current = 10
    last_data = None
    current_data = np.loadtxt(dist+str(current))
    done = False
    while current < 160:
        current += 5
        last_data = current_data
        try:
            current_data = np.loadtxt(dist+str(current))
        except:
            break
        result = rmse(current_data,last_data)
        rmse_vals.append(result)
        print "RMSE (%s) %s <=> %s: %.02f" % (dist_dict[dist],str(current),str(current-5),result)




