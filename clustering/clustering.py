from __future__ import print_function
import sys
import numpy as np
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel


def list_ratings(lists, count):
	result = np.zeros(count)
	for l in lists:
		result[l[0]] = l[1]
	return result

if __name__ == "__main__":
	 spark = SparkSession.builder.appName("Clustering").getOrCreate()
	 data = spark.read.json("../dataset-extraction/thousand.json", multiLine=True)
	 data_rdd = data.select('user_id','business_id','stars').rdd.map(tuple)
	 user_map = data_rdd.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
	 business_distinct = data_rdd.map(lambda x: x[1]).distinct()
	 business_count = business_distinct.count() 
	 business_map = business_distinct.zipWithIndex().collectAsMap()

	 # user_bc = sc.broadcast(user_map)
	 # business_bc = sc.broadcast(business_map)

	 data_index = data_rdd.map(lambda x: (user_map.get(x[0]), business_map.get(x[1]), float(x[2])))
	 user_group = data_index.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(list)
	 user_stars = user_group.map(lambda x: (x[0],list_ratings(x[1], business_count)))
	 clusters = KMeans.train(user_stars.map(lambda x: x[1]), 3, maxIterations=10, initializationMode="random")

	 def error(x):
	 	centroid = clusters.centers[clusters.predict(x)]
	 	return sqrt(sum([x**2 for x in (x - centroid)]))
	 	
	 # compute Within Set Sum of Squared Error (WSSSE)
	 wsse = user_stars.map(lambda x: error(x[1])).reduce(lambda x, y: x + y)
	 print("Within Set Sum of Squared Error = " + str(wsse))

	 user_clusters = user_stars.map(lambda x: (x[0], clusters.predict(x[1])))
	 user_comparison = user_group.join(user_clusters)




