from __future__ import print_function
import sys
import numpy as np
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import json
import time

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Visualization").getOrCreate()
    sc = spark.sparkContext

	# Start time
    start_time = time.time()

    data_original = spark.read.json("./converged_stars500.json", multiLine=True)
    data = data_original.select('user_id', 'business_id', 'stars', 'Adj_Stars').rdd.map(tuple)
    user_average = data.map(lambda (user, business, star, astar): (user, (star, astar, 1))).reduceByKey(lambda (s_x, as_x, c_x), (s_y, as_y, c_y) : (s_x + s_y, as_x + as_y, c_x + c_y)).map(lambda (user, (s, ads, c)) : (user, (float(s)/float(c), float(ads)/float(c))))
    # for x in user_average.take(10):
    #     print(x)

    business_average = data.map(lambda (user, business, star, astar): (business, (star, astar, 1))).reduceByKey(lambda (s_x, as_x, c_x), (s_y, as_y, c_y) : (s_x + s_y, as_x + as_y, c_x + c_y)).map(lambda (business, (s, ads, c)) : (business, (float(s)/float(c), float(ads)/float(c))))

    aggregate_data = data.map(lambda (u, b, s, ads) : (u, (b, s, ads))).join(user_average).map(lambda (u, ((b, s, ads), (us, uads))) : (b, ((u, us, uads), (s, ads)))).join(business_average).map(lambda (b, (((u, us, uads), (s, ads)), (bs, bads))) : ((u, us, uads), (b, bs, bads), (s, ads)))

    # diff_data = aggregate_data.map(lambda ((u, us, uads), (b, bs, bads), (s, ads)) : ((us - bs, uads - bads), (s - us, ads - uads)))
    # diff_data = aggregate_data.map(lambda ((u, us, uads), (b, bs, bads), (s, ads)) : ((us - bs, uads - bads), (s - bs, ads - bads)))
    # diff_data = aggregate_data.map(lambda ((u, us, uads), (b, bs, bads), (s, ads)) : ((s - us, ads - uads), (s - bs, ads - bads)))
    diff_original = aggregate_data.map(lambda ((u, us, uads), (b, bs, bads), (s, ads)) : (us - bs, s - us, s - bs))
    diff_adjusted = aggregate_data.map(lambda ((u, us, uads), (b, bs, bads), (s, ads)) : (uads - bads, ads - uads, ads - bads))

    diff_original_plot = diff_original.collect()
    diff_adjusted_plot = diff_adjusted.collect()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.subplot(1, 2, 1)
    ax.scatter(map(lambda (ds_one, ds_two, ds_three): ds_one, diff_original_plot), map(lambda (ds_one, ds_two, ds_three): ds_two, diff_original_plot), map(lambda (ds_one, ds_two, ds_three): ds_three, diff_original_plot), label='Original Stars')
    # plt.title('Original Stars')
    # plt.xlabel('star - business_average')
    ax.set_ylabel('star - user_average')
    ax.set_xlabel('user_average - business_average')
    ax.set_zlabel('star - business_average')

    # plt.subplot(1, 2, 2)
    ax.scatter(map(lambda (ds_one, ds_two, ds_three): ds_one, diff_adjusted_plot), map(lambda (ds_one, ds_two, ds_three): ds_two, diff_adjusted_plot), map(lambda (ds_one, ds_two, ds_three): ds_three, diff_adjusted_plot), label='Adjusted Stars')
    # plt.title('Adjusted Stars')
    # plt.xlabel('star - business_average')
    # plt.ylabel('user_average - business_average')
    # plt.ylabel('star - user_average')

    ax.legend()

    plt.show()

    # End time
    end_time = time.time()
    print("Total execution time: " + str(end_time - start_time))
