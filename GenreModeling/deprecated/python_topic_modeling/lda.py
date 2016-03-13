import graphlab as gl
from graphlab.toolkits._main import ToolkitError
import numpy as np
import time
import datetime

gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 8)
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_GRAPH_LAMBDA_WORKERS', 8)
#gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY',100000000000)
#gl.set_runtime_config('GRAPHLAB_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE',100000000000)

raw_docs = gl.SArray("LDA_vectors/")
vocab_idx = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    vocab_idx[int(line[1])] = line[0]

def formatter(row):
    row = eval(row)
    row = dict(zip([vocab_idx[term] for term in row[1][1]],row[1][2]))
    return row

docs = raw_docs.filter(lambda x: x!="").apply(formatter)
docs.save("doc_array",format='binary')

docs = gl.SArray("doc_array")

train,test = gl.SFrame(docs).random_split(0.9,seed=99)
train = gl.text_analytics.tf_idf(train['X1'])
test = gl.text_analytics.tf_idf(test['X1'])
#data = {'tfidf':(train_tfidf,test_tfidf),'tf':(train,test)}


n_iter = 50

with open('log_tfidf','w') as log:
    log.write('\t'.join(['k','alpha','beta','perplexity'])+'\n')
    for k in np.arange(90,301,5,dtype=float):
        #for alpha in [i/k for i in [1,5,10,25,50,75,100]]:
        #for beta in [0.01, 0.05, 0.1, 0.5, 1.0]:
        #for norm in ('tfidf','tf'):
            #train_data,test_data = data[norm]
            #print 'running model for k=%s,alpha=%s,beta=%s' % (k,alpha,beta)
            print 'running model for k=%s' % k
            try:
                overall_start = time.time()
                start = time.time()
                #topic_model = gl.topic_model.create(train_data,num_topics=k,num_iterations=n_iter,alpha=alpha,beta=beta,method='cgs')
                #for n_iter in (60,70):
                topic_model = gl.topic_model.create(train,num_topics=k,num_iterations=n_iter,method='cgs')
                print 'model run complete in %s' % str(datetime.timedelta(seconds=(time.time()-start)))
                start = time.time()
                #perplexity = topic_model.evaluate(train_data, test_data)['perplexity']
                perplexity = topic_model.evaluate(test)['perplexity']
                print 'perplexity calculated in %s' % str(datetime.timedelta(seconds=(time.time()-start)))
                #result = '\t'.join(map(str,[k,norm,topic_model.get('alpha'),topic_model.get('beta'),perplexity]))
                result = '\t'.join(map(str,[k,topic_model.get('alpha'),topic_model.get('beta'),perplexity]))
                print result
                log.write(result+'\n')
                log.flush()
                print 'overall time for this test: %s' % str(datetime.timedelta(seconds=(time.time()-overall_start)))
            except ToolkitError as e:
                if "Assertion failed" in str(e):
                    continue
                else:
                    raise(e)






    top_terms_by_topic = topic_model.get_topics(num_words=TOP_TERMS,output_type='topic_probabilities').to_dataframe()
    top_terms_by_topic['Rank'] = (-1*top_terms_by_topic).groupby('topic')['score'].transform(np.argsort)
    top_terms_by_topic['New_str'] = top_terms_by_topic['word'] + top_terms_by_topic['score'].apply(' ({0:.2f})'.format)
    new = top_terms_by_topic.sort_values(by=['Rank', 'score'])[['New_str', 'topic','Rank']]
    print new.pivot(index='Rank', values='New_str', columns='topic')

    topic_model = gl.topic_model.create(docs,num_topics=10,num_iterations=20)
    print topic_model.evaluate(docs)

    top_terms_by_topic = topic_model.get_topics(num_words=TOP_TERMS,output_type='topic_probabilities').to_dataframe()
    top_terms_by_topic['Rank'] = (-1*top_terms_by_topic).groupby('topic')['score'].transform(np.argsort)
    top_terms_by_topic['New_str'] = top_terms_by_topic['word'] + top_terms_by_topic['score'].apply(' ({0:.2f})'.format)
    new = top_terms_by_topic.sort_values(by=['Rank', 'score'])[['New_str', 'topic','Rank']]
    print new.pivot(index='Rank', values='New_str', columns='topic')
