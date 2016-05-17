"""
Run the "random" LDA model
"""
import graphlab as gl
import numpy as np

gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 32)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)

np.random.seed(999)

docs = gl.SArray('doc_array')
user_playcounts = {str(i):cnt for i,cnt in enumerate(docs.apply(lambda x: int(sum(x.values()))))}

artist_playcounts = {line.strip().split('\t')[1]:0 for line in open('artist_ids')}

for d in docs:
    for k in d:
        artist_playcounts[k]+=int(d[k])

random_data = {}

for i,(artist,playcount) in enumerate(artist_playcounts.iteritems()):
    print i,artist,playcount
    random_data[artist] = {}
    for j,listen in enumerate(xrange(playcount)):
        user = np.random.choice(user_playcounts.keys())
        random_data[artist][user] = random_data[artist].get(user,0)+1
        user_playcounts[user]-=1
        if user_playcounts[user]==0:
            user_playcounts.pop(user)

new_docs = gl.SArray(random_data.values())
new_docs.save('random_doc_array',format='binary')


topic_model = gl.topic_model.create(new_docs,num_topics=190,num_iterations=100,method='cgs')
artist_topic_probs = np.array(topic_model.predict(new_docs,output_type='probabilities'))
np.save('random_lda_features.npy',artist_topic_probs)
