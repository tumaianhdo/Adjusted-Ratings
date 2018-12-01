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
import time



if __name__ == "__main__":
	spark = SparkSession.builder.appName("Clustering").getOrCreate()
	sc = spark.sparkContext

	# Start time
	start_time = time.time()

	# data_sentiment = spark.read.json("./google_results.json", multiLine=True)
	# data_sentiment_rdd = data_sentiment.select('user_id','business_id','sentiment-google').rdd.map(tuple)

	# data_normalized = spark.read.json("./user_normalizedStars.json", multiLine=True)
	# data_normalized_rdd = data_normalized.select('user_id', 'business_id', 'Adj_Stars').rdd.map(tuple)

	# # data_normalized = spark.read.json("./thousand.json", multiLine=True)
	# # data_normalized_rdd = data_normalized.select('user_id', 'business_id', 'stars').rdd.map(tuple)

	# data_sentiment_refined = data_sentiment_rdd.map(lambda (user, business, star): ((user, business), star))
	# data_normalized_refined = data_normalized_rdd.map(lambda (user, business, star): ((user, business), star))

	# data = data_normalized_refined.join(data_sentiment_refined)

	data_original = spark.read.json("./converged_stars500.json", multiLine=True)
	data_original_rdd = data_original.select('user_id', 'business_id', 'Adj_Stars').rdd.map(tuple)
	data = data_original_rdd.map(lambda (user, business, star): ((user, business), star)).partitionBy(8)
	print("Number of partitions : " + str(data.getNumPartitions()))

	user_distinct = data.map(lambda ((user, business), star): user).distinct()
	print("Number of users: " + str(user_distinct.count()))
	user_map = user_distinct.zipWithIndex().collectAsMap()
	business_distinct = data.map(lambda ((user, business), star): business).distinct()
	business_count = business_distinct.count()
	print("Number of businesses: " + str(business_count))
	business_map = business_distinct.zipWithIndex().collectAsMap()
	user_bc = sc.broadcast(user_map)
	business_bc = sc.broadcast(business_map)

	reverse_user_map = {v:k for (k,v) in user_map.items()}
	reverse_user_bc = sc.broadcast(reverse_user_map)

	data_index = data.map(lambda ((user, business), star): (user_bc.value[user], (business_bc.value[business], float(star))))
	user_group = data_index.groupByKey()

	def rating_vector(lbrs):
		vector = {}
		for (business, star) in lbrs:
			vector[business] = star
		return Vectors.sparse(business_count, vector)

	user_stars = user_group.map(lambda (user, lbrs): (user, rating_vector(lbrs)))
	star_arrays = user_stars.map(lambda (user, star_array): star_array)
	indexes_users = user_stars.map(lambda (user, star_array): user).zipWithIndex().map(lambda (user, index): (index, user))

	# star_mat = RowMatrix(star_arrays)
	# pca = star_mat.computePrincipalComponents(2)
	# projected_stars = star_mat.multiply(pca).rows
	# for x in projected_stars.collect():
	# 	print(x)


	# num_clusters = 5
	# clusters = KMeans.train(projected_stars, num_clusters, maxIterations=20, initializationMode="k-means||")
	# cost = clusters.computeCost(projected_stars)
	# print("K-means Clustering Cost = " + str(cost))

	# clustered_stars = clusters.predict(projected_stars)
	# cluster_counts = clustered_stars.map(lambda cluster: (cluster, 1)).reduceByKey(lambda x, y: x + y)
	# for x in cluster_counts.collect():
	# 	print(x)

	num_clusters = 5
	clusters = KMeans.train(star_arrays, num_clusters, maxIterations=20, initializationMode="kmeans||")
	cost = clusters.computeCost(star_arrays)
	print("K-means Clustering Cost = " + str(cost))

	clustered_stars = clusters.predict(star_arrays)
	cluster_counts = clustered_stars.map(lambda cluster: (cluster, 1)).reduceByKey(lambda x, y: x + y)
	for x in cluster_counts.collect():
		print(x)

	# indexes_projected_stars = projected_stars.zipWithIndex().map(lambda (coordinates, index): (index, coordinates))
	# indexes_clustered_stars = clustered_stars.zipWithIndex().map(lambda (cluster, index) : (index, cluster))
	#
	# projected_stars_clusters = indexes_projected_stars.join(indexes_clustered_stars)
	# clusters_project_stars = projected_stars_clusters.map(lambda (index, (coordinates, cluster)) : (cluster, coordinates))
	# clusters_coordinates = clusters_project_stars.collect()
	#
	# for c in range(num_clusters):
	# 	group = filter(lambda (cluster, coordinates) : cluster == c, clusters_coordinates)
	# 	plt.plot(map(lambda (cluster, coordinates): coordinates[0], group), map(lambda (cluster, coordinates): coordinates[1], group),'o')
	#
	# plt.show()

	# predict_data = user_stars.map(lambda (user, stars): (clusters.predict(stars), user))
	# cluster_users = predict_data.map(lambda (cluster, user): (cluster, reverse_user_bc.value[user])).groupByKey().mapValues(list)
	# result_data = cluster_counts.join(cluster_users).map(lambda (cluster, attributes): (cluster, { "size": attributes[0], "user": attributes[1] }))
	#
	# with open("./result.json", "w") as f:
	# 	json.dump(result_data.collectAsMap(),f)

	# End time
	end_time = time.time()
	print("Total execution time: " + str(end_time - start_time))
