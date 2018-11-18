from __future__ import print_function
import sys
import numpy as np
from numpy import array
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
import matplotlib.pyplot as plt
import json



if __name__ == "__main__":
	spark = SparkSession.builder.appName("Clustering").getOrCreate()
	sc = spark.sparkContext

	data_sentiment = spark.read.json("./google_results.json", multiLine=True)
	data_sentiment_rdd = data_sentiment.select('user_id','business_id','sentiment-google').rdd.map(tuple)

	data_normalized = spark.read.json("./user_normalizedStars.json", multiLine=True)
	data_normalized_rdd = data_normalized.select('user_id', 'business_id', 'Adj_Stars').rdd.map(tuple)

	# data_normalized = spark.read.json("./thousand.json", multiLine=True)
	# data_normalized_rdd = data_normalized.select('user_id', 'business_id', 'stars').rdd.map(tuple)

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
	user_group = data_index.groupByKey()


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

	# def list_ratings(lists):
	# 	result = np.zeros(business_count*2)
	# 	average0 = np.mean(map(lambda (business, star): star[0], lists))
	# 	average1 = np.mean(map(lambda (business, star): star[1], lists))
	# 	for (business, star) in lists:
	# 		result[business] = star[0] - average0
	# 		result[business + business_count] = star[1] - average1
	# 	return result

	# user_stars = user_group.map(lambda (user, lbrs): (user, list_ratings(lbrs)))

	def rating_vector(lbrs):
		vector = {}
		for (business, star) in lbrs:
			vector[business] = star[0]
			vector[business + business_count] = star[1]
		return Vectors.sparse(business_count * 2, vector)

	user_stars = user_group.map(lambda (user, lbrs): (user, rating_vector(lbrs)))
	star_arrays = user_stars.map(lambda (user, star_array): star_array)
	num_clusters = 5
	clusters = KMeans.train(star_arrays, num_clusters, maxIterations=20, initializationMode="k-means||")
	cost = clusters.computeCost(star_arrays)
	print("K-means Clustering Cost = " + str(cost))

	
	predict_count = predict_data.map(lambda (cluser, user): (cluser, 1)).reduceByKey(lambda x,y: x+y)
	for x in predict_count.collect():
		print(x)

	

	star_mat = RowMatrix(star_arrays)
	pca = star_mat.computePrincipalComponents(2)
	projected_stars = star_mat.multiply(pca).rows
	# for x in projected_stars.collect():
	# 	print(x) 

	

	clustered_stars = clusters.predict(star_arrays)
	cluster_counts = clustered_stars.map(lambda cluster: (cluster, 1)).reduceByKey(lambda x, y: x + y)
	for x in cluster_counts.collect():
		print(x)

	indexes_projected_stars = projected_stars.zipWithIndex().map(lambda (coordinates, index): (index, coordinates))
	indexes_clustered_stars = clustered_stars.zipWithIndex().map(lambda (cluster, index) : (index, cluster))
	projected_stars_clusters = indexes_projected_stars.join(indexes_clustered_stars).map(lambda (index, (coordinates, cluster)) : (cluster, coordinates))
	clusters_coordinates = projected_stars_clusters.collect()

	for c in range(num_clusters):
		group = filter(lambda (cluster, coordinates) : cluster == c, clusters_coordinates)
		plt.plot(map(lambda (cluster, coordinates): coordinates[0], group), map(lambda (cluster, coordinates): coordinates[1], group),'o')
	
	plt.show()	

	predict_data = user_stars.map(lambda (user, stars): (clusters.predict(stars), user))
	cluster_users = predict_data.map(lambda (cluster, user): (cluster, reverse_user_bc.value[user])).groupByKey().mapValues(list)
	result_data = cluster_counts.join(cluster_users).map(lambda (cluster, attributes): (cluster, { "size": attributes[0], "user": attributes[1] }))

	with open("./result.json", "w") as f:
		json.dump(result_data.collectAsMap(),f)

	
	 



