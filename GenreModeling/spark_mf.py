# IPYTHON=1 spark-1.6.0-bin-hadoop2.6/bin/pyspark --driver-memory=110G

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, ArrayType, MapType, FloatType
import time,datetime
import numpy as np
import os


d = 'MF_tests/'

k_range = np.arange(10,201,10)
split = True
#alpha_range = [0.0001,0.001,0.01,0.05,1.0,5,10]
#reg_range =[1e-8,1e-04,0.01,0.1,0.5,1.0]


raw_data = sc.textFile("mf_format.txt").map(lambda row: [int(val) for val in row.strip().split(',')])

if split:
    train,test = raw_data.randomSplit(weights=[0.9,0.1],seed=99)
    ratings = train.map(lambda row: Rating(row[0],row[1],row[2])).persist()
    testData = test.map(lambda p: (p[0], p[1]))
    base_model_name = d+'model_split_'
else:
    ratings = raw_data.map(lambda row: Rating(row[0],row[1],row[2])).persist()
    base_model_name = d+'model_'


with open(d+'log_rmse','a') as fout:
    for k in k_range:
        start = time.time()

        if os.path.exists(base_model_name+str(k)):
            continue
            #model = MatrixFactorizationModel.load(sc,d+'model_split_%s' % k)

        else:
            model = ALS.trainImplicit(ratings,rank=k,iterations=5,alpha=0.01,nonnegative=False)
            model.save(sc,base_model_name+str(k))
            if split:
                predictions = model.predictAll(testData).map(lambda r: ((r[0], r[1]), r[2]))
                ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
                RMSE = np.sqrt(ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
                print("Mean Squared Error = " + str(RMSE))
                fout.write('\t'.join(map(str,[k,RMSE]))+'\n')
                fout.flush()
            else:
                artist_features = np.array(model.productFeatures().sortByKey().map(lambda row: row[1]).collect())
                np.save(d+"features_{}".format(k))







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
