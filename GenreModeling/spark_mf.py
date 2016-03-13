from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, ArrayType, MapType, FloatType
import time,datetime
import numpy as np
import graphlab as gl
import pandas as pd
import os
gl.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 32)

d = 'MF_tests/'
N=100

k_range = np.arange(10,201,10)
#alpha_range = [0.0001,0.001,0.01,0.05,1.0,5,10]
#reg_range =[1e-8,1e-04,0.01,0.1,0.5,1.0]

artist_indices = {}
id_idx = {}
for line in open('vocab_idx'):
    line = line.strip().split('\t')
    id_idx[int(line[1])] = line[0]
    artist_indices[line[0]] = int(line[1])
lastfm_data = np.zeros((112312,100),dtype=int)
lastfm_data.fill(-1)
with open('lastfm_top_similar_artists') as fin:
    for line in fin:
        try:
            a,top100 = line.strip().split('\t')
        except ValueError:
            continue
        aid = artist_indices.get(a)
        if aid is not None:
            for i,sim in enumerate(top100.split()):
                lastfm_data[aid,i] = artist_indices.get(sim,-1)


# base_linear_weights = np.arange(1,101,1)[::-1]
# base_linear_weights_sum = float(base_linear_weights.sum())
# base_log_weights = 10./np.arange(1,101,1)
# base_log_weights_sum = float(base_log_weights.sum())
def proc_fm(row,weight=None,comparison='all',alignment=None,threshold=50,top=100):
    fm = np.array(row['fm'])
    if weight == None:
        if comparison == 'all':
            return len(set(row['current'][:top]).intersection(fm[:top]))/float(top)
        elif comparison == 'possible':
            fm = set(fm[:top])
            fm.discard(-1)
            if len(fm)==0:
                return None
            return len(set(row['current'][:top]).intersection(fm))/float(len(fm))
        elif comparison == 'threshold':
            fm = set(fm[:top])
            fm.discard(-1)
            if len(fm)<threshold:
                return None
            return len(set(row['current'][:top]).intersection(fm))/float(len(fm))
    else:
        if weight == 'linear':
            weights = base_linear_weights
            weight_sum = base_linear_weights_sum

        elif weight == 'log':
            weights = base_log_weights
            weight_sum = base_log_weights_sum

        if alignment is None:
            intersection = set(row['current'][:100]).intersection(set(fm))
            indices = [i for i,val in enumerate(row['fm']) if val in intersection]
            if comparison == 'all':
                return weights[indices].sum() / weight_sum
            elif comparison == 'possible':
                if np.all(fm==-1):
                    return None
                return weights[indices].sum() / float(weights[fm!=-1].sum())
        else:
            pass


combined = gl.SFrame({'fm':lastfm_data})

#raw_data = sc.textFile("mf_format.txt").map(lambda row: [int(val) for val in row.strip().split(',')])
#ratings = raw_data.map(lambda row: Rating(row[0],row[1],row[2])).persist()

if k_range[0]==10:
    last = None
else:
    last = gl.SArray(d+'knn_%s' % (k_range[0]-10))

top=100
with open(d+'log_threhsold_75','a') as fout:#,open(d+'log_rmse','a') as rmseout:
    for k in k_range:
        start = time.time()
        ddir = d+'knn_%s' % (k)
        if os.path.exists(ddir):
            topN = gl.SArray(ddir)
            #pass
        else:

            if os.path.exists(d+'model_%s' % k):
                model = MatrixFactorizationModel.load(sc,d+'model_%s' % k)
            else:
                model = ALS.trainImplicit(ratings,rank=k,iterations=5,alpha=0.01,nonnegative=False)
                model.save(sc,d+'model_%s' % k)
                # predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
                # ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
                # RMSE = np.sqrt(ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
                # print("Mean Squared Error = " + str(RMSE))
                # rmseout.write('\t'.join(map(str,[k,alpha,reg,RMSE]))+'\n')
                # rmseout.flush()
            artist_features = np.array(model.productFeatures().sortByKey().map(lambda row: row[1]).collect())
            sf = gl.SFrame(artist_features).add_row_number()
            result = gl.nearest_neighbors.create(dataset=sf,label='id',features=['X1'],distance='cosine').similarity_graph(k=100,output_type='SFrame')
            topN = result[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i]  for i in xrange(1,100+1)])
            topN.save(ddir)

        combined['current'] = topN
        overlap = combined.apply(lambda row: proc_fm(row,comparison='threshold',threshold=75)).dropna()
        summary = pd.Series(overlap).describe()
        fout.write('\t'.join(map(str,['fm',k]+list(summary)))+'\n')
        fout.flush()
        print 'Model run complete in %s' % str(datetime.timedelta(seconds=(time.time()-start)))

        # if last is not None:
        #     combined_prev = gl.SFrame({'current':topN, 'prev':last})
        #     overlap = combined_prev.apply(lambda row: len(set(row['current'][:top]).intersection(set(row['prev'][:top])))/float(top))
        #     summary = pd.Series(overlap).describe()
        #     fout.write('\t'.join(map(str,['prev',k]+list(summary)))+'\n')
        #     fout.flush()
        # last = topN




"""
raw_data = sc.textFile("mf_format.txt").map(lambda row: [int(val) for val in row.strip().split(',')])
alpha = 0.01
eps = 0.1
k=50
rating_transforms = [lambda x: x, lambda x: min(5,1+np.log10(x)), lambda x: min(10,np.log(1+(x/eps)))]

with open(d+'test','a') as fout:
    for rating_transform in rating_transforms:
        ratings = raw_data.map(lambda row: Rating(row[0],row[1],rating_transform(row[2]))).persist() # .filter(lambda row: row[2]>10)
        model = ALS.trainImplicit(ratings,rank=k,iterations=5,alpha=alpha,nonnegative=False)
        artist_features = np.array(model.productFeatures().sortByKey().map(lambda row: row[1]).collect())
        sf = gl.SFrame(artist_features).add_row_number()
        result = gl.nearest_neighbors.create(dataset=sf,label='id',features=['X1'],distance='cosine').similarity_graph(k=100,output_type='SFrame')
        topN = result[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i]  for i in xrange(1,100+1)])
        combined['current'] = topN
        overlap = combined.apply(lambda row: proc_fm(row))
        summary = pd.Series(overlap).describe()
        fout.write('\t'.join(map(str,list(summary)))+'\n')
        fout.flush()

rating_transforms = [lambda x: x, lambda x: min(4,np.log10(x)), lambda x: min(10,np.log(1+(x/eps)))]
raw_data = raw_data.filter(lambda row: row[2]>5)

with open(d+'test','a') as fout:
    for rating_transform in rating_transforms:
        ratings = raw_data.map(lambda row: Rating(row[0],row[1],rating_transform(row[2]))).persist() # .filter(lambda row: row[2]>10)
        model = ALS.trainImplicit(ratings,rank=k,iterations=5,alpha=alpha,nonnegative=False)
        artist_features = np.array(model.productFeatures().sortByKey().map(lambda row: row[1]).collect())
        sf = gl.SFrame(artist_features).add_row_number()
        result = gl.nearest_neighbors.create(dataset=sf,label='id',features=['X1'],distance='cosine').similarity_graph(k=100,output_type='SFrame')
        topN = result[['query_label','reference_label','rank']].unstack(('rank','reference_label'),new_column_name='knn').sort('query_label').apply(lambda row: [row['knn'][i]  for i in xrange(1,100+1)])
        combined['current'] = topN
        overlap = combined.apply(lambda row: proc_fm(row))
        summary = pd.Series(overlap).describe()
        fout.write('\t'.join(map(str,list(summary)))+'\n')
        fout.flush()
"""
