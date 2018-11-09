from __future__ import print_function
import sys
import numpy as np
from numpy import array
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel
import matplotlib.pyplot as plt


def list_ratings(lists, count):
	result = np.zeros(count*2)
	average0 = np.mean(map(lambda (business, star): star[0], lists))
	average1 = np.mean(map(lambda (business, star): star[1], lists))
	for (business, star) in lists:
		result[business] = star[0] - average0
		result[business + count] = star[1] - average1
	return result

if __name__ == "__main__":
	 spark = SparkSession.builder.appName("Clustering").getOrCreate()
	 sc = spark.sparkContext

	 data_sentiment = spark.read.json("./google_results.json", multiLine=True)
	 data_sentiment_rdd = data_sentiment.select('user_id','business_id','sentiment-google').rdd.map(tuple)

	 # data_normalized = spark.read.json("./user_normalizedStars.json", multiLine=True)
	 # data_normalized_rdd = data_normalized.select('user_id', 'business_id', 'Adj_Stars').rdd.map(tuple)

	 data_normalized = spark.read.json("./thousand.json", multiLine=True)
	 data_normalized_rdd = data_normalized.select('user_id', 'business_id', 'stars').rdd.map(tuple)

	 data_sentiment_refined = data_sentiment_rdd.map(lambda (user, business, star): ((user, business), star))
	 data_normalized_refined = data_normalized_rdd.map(lambda (user, business, star): ((user, business), star))

	 data = data_normalized_refined.join(data_sentiment_refined)

	 user_map = data.map(lambda (ub, stars): ub[0]).distinct().zipWithIndex().collectAsMap()
	 business_distinct = data.map(lambda (ub,stars): ub[1]).distinct()
	 business_count = business_distinct.count() 
	 business_map = business_distinct.zipWithIndex().collectAsMap()
	 user_bc = sc.broadcast(user_map)
	 business_bc = sc.broadcast(business_map)

	 reverse_user_map = {v:k for (k,v) in user_map.items()}
	 reverse_user_bc = sc.broadcast(reverse_user_map) 

	 data_index = data.map(lambda (ub, stars): (user_bc.value[ub[0]], (business_bc.value[ub[1]], (float(stars[0]), float(stars[1])))))
	 user_group = data_index.groupByKey().mapValues(list)
	 user_stars = user_group.map(lambda (user, lists): (user, list_ratings(lists, business_count)))

	 # num_clusters = 5
	 # clusters = KMeans.train(data_index.map(lambda x: array([float(x[1]), x[2]])), num_clusters, maxIterations=20, initializationMode="random")	
	
	 # def error(x):
	 # 	centroid = clusters.centers[clusters.predict(x)]
	 # 	return sqrt(sum([x**2 for x in (x - centroid)]))
		
	 # # compute Within Set Sum of Squared Error (WSSSE)
	 # wsse = user_stars.map(lambda x: error(x[1])).reduce(lambda x, y: x + y)
	 # print("Within Set Sum of Squared Error = " + str(wsse))

	 # predict_data = data_index.map(lambda (user, business, star): [user, business, star, clusters.predict(array([float(business), star]))])
	 # predict_data_list = predict_data.collect()

	 # for cluster in range(num_clusters):
		# group = g1 = filter(lambda x: x[3] == cluster, predict_data_list)
		# plt.plot(map(lambda x: x[2], group), map(lambda x: x[0], group),'o')

	# plt.show()	

	 num_clusters = 5
	 clusters = KMeans.train(user_stars.map(lambda (user, star_array): star_array), num_clusters, maxIterations=20, initializationMode="random")

	 predict_data = user_stars.map(lambda (user, stars): (clusters.predict(stars), user))
	 predict_count = predict_data.map(lambda (cluser, user): (cluser, 1)).reduceByKey(lambda x,y: x+y)
	 cluster_user = predict_data.map(lambda (cluster, user): (cluster, reverse_user_bc.value[user])).groupByKey().mapValues(list)

	 for x in predict_count.collect():
	 	print(x)
	 for x in cluster_user.collect():
	 	print(x)
	 



