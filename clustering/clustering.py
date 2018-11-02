from __future__ import print_function
import sys
import numpy as np
from numpy import array
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel
import matplotlib.pyplot as plt


def list_ratings(lists, count):
	result = np.zeros(count)
	for l in lists:
		result[l[0]] = l[1]
	return result

if __name__ == "__main__":
	spark = SparkSession.builder.appName("Clustering").getOrCreate()
	sc = spark.sparkContext

	data = spark.read.json("../sentiment-api-results/google_results.json", multiLine=True)
	data_rdd = data.select('user_id','business_id','sentiment-google').rdd.map(tuple)

	user_map = data_rdd.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
	business_distinct = data_rdd.map(lambda x: x[1]).distinct()
	business_count = business_distinct.count() 
	business_map = business_distinct.zipWithIndex().collectAsMap()
	user_bc = sc.broadcast(user_map)
	business_bc = sc.broadcast(business_map)

	data_index = data_rdd.map(lambda x: (user_bc.value[x[0]], business_bc.value[x[1]], float(x[2])))
	data_refined = data_index.map(lambda x: array([float(x[0]), float(x[1]), x[2]]))
	# user_group = data_index.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(list)
	# user_stars = user_group.map(lambda x: (x[0],list_ratings(x[1], business_count)))

	def error(x):
		centroid = clusters.centers[clusters.predict(x)]
		return sqrt(sum([x**2 for x in (x - centroid)]))

	 
	# for ncluster in range(50, 500, 50):
	# 	clusters = KMeans.train(data_refined, ncluster, maxIterations=500, initializationMode="random")
	# 	# compute Within Set Sum of Squared Error (WSSSE)
	# 	wsse = data_refined.map(lambda x: error(x)).reduce(lambda x, y: x + y)
	# 	group_count = data_index.map(lambda (user, business, star): (clusters.predict(array([float(user), float(business), star])), 1)).reduceByKey(lambda x,y: x + y)
	# 	print("ncluster = " + str(ncluster) + " WSSE = " + str(wsse))
	# 	for x in group_count.collect():
	# 		print(x)

	# for ncluster in range(5, 50, 5):
	# 	clusters = KMeans.train(user_stars.map(lambda x: x[1]), ncluster, maxIterations=500, initializationMode="random")
	# 	# compute Within Set Sum of Squared Error (WSSSE)
	# 	wsse = user_stars.map(lambda x: error(x[1])).reduce(lambda x, y: x + y)
	# 	group_count = user_stars.map(lambda (user, stars): (clusters.predict(stars), 1)).reduceByKey(lambda x,y: x + y)
	# 	print("ncluster = " + str(ncluster) + " WSSE = " + str(wsse))
	# 	for x in group_count.collect():
	# 		print(x)

	num_clusters = 5
	clusters = KMeans.train(data_refined, num_clusters, maxIterations=500, initializationMode="random")

	# compute Within Set Sum of Squared Error (WSSSE)
	wsse = data_refined.map(lambda x: error(x)).reduce(lambda x, y: x + y)
	group_count = data_index.map(lambda (user, business, star): (clusters.predict(array([float(user), float(business), star])), 1)).reduceByKey(lambda x,y: x + y)
	print("WSSE = " + str(wsse))
	print("Items distribution across clusters")
	print(",".join(group_count.map(lambda (cluster, count): str(count)).collect()))

	predict_data = data_index.map(lambda (user, business, star): [user, business, star, clusters.predict(array([float(user), float(business), star]))])
	predict_data_list = predict_data.collect()

	for cluster in range(num_clusters):
		group = filter(lambda x: x[3] == cluster, predict_data_list)
		plt.plot(map(lambda x: x[2], group), map(lambda x: x[0], group),'o')

	plt.show()



