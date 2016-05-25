

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, ArrayType, MapType, FloatType
import time,datetime
import numpy as np
import os


d = 'NMF_tests_10iter/'
n_iter = 10

k_range = np.arange(200,201,10)
split = False
#alpha_range = [0.0001,0.001,0.01,0.05,1.0,5,10]
#reg_range =[1e-8,1e-04,0.01,0.1,0.5,1.0]


raw_data = sc.textFile("mf_format.txt").map(lambda row: [int(val) for val in row.strip().split(',')])

if split:
    rand_a,rand_b = raw_data.randomSplit(weights=[0.5,0.5],seed=99).persist()
    ratings_a = rand_a.map(lambda row: Rating(row[0],row[1],row[2])).persist()
    ratings_b = rand_b.map(lambda row: Rating(row[0],row[1],row[2])).persist()
else:
    ratings = raw_data.map(lambda row: Rating(row[0],row[1],row[2])).persist()
    base_model_name = d+'model_'


#with open(d+'log_rmse','a') as fout:
for k in k_range:
    start = time.time()

    if split:
        model_a = ALS.trainImplicit(ratings_a,rank=k,iterations=n_iter,alpha=0.01,nonnegative=True)
        model_b = ALS.trainImplicit(ratings_a,rank=k,iterations=n_iter,alpha=0.01,nonnegative=True)
        model_a.save(sc,'model_rand_a_'+str(k))
        model_b.save(sc,'model_rand_b_'+str(k))
        artist_features_a = np.array(model_a.productFeatures().sortByKey().map(lambda row: row[1]).collect())
        np.save(d+"features_rand_a_{}".format(k))
        artist_features_b = np.array(model_b.productFeatures().sortByKey().map(lambda row: row[1]).collect())
        np.save(d+"features_rand_b_{}".format(k))
    else:
        model = ALS.trainImplicit(ratings,rank=k,iterations=n_iter,alpha=0.01,nonnegative=True)
        model.save(sc,d+'model_'+str(k))
        artist_features = np.array(model.productFeatures().sortByKey().map(lambda row: row[1]).collect())
        np.save(d+"features_{}".format(k),artist_features)
        user_features = np.array(model.userFeatures().sortByKey().map(lambda row: row[1]).collect())
        np.save(d+"user_features_{}".format(k),user_features)





