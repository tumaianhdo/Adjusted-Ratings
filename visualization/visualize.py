from __future__ import print_function
import sys
import numpy as np
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import json
import time

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Visualization").getOrCreate()
    sc = spark.sparkContext

	# Start time
    start_time = time.time()

    data_original = spark.read.json("./converged_stars.json", multiLine=True)
    data = data_original.select('user_id', 'business_id', 'stars', 'Adj_Stars').rdd.map(tuple)
    user_average = data.map(lambda (user, business, star, astar): (user, (star, astar, 1))).reduceByKey(lambda (s_x, as_x, c_x), (s_y, as_y, c_y) : (s_x + s_y, as_x + as_y, c_x + c_y)).map(lambda (user, (s, ads, c)) : (user, (float(s)/float(c), float(ads)/float(c))))
    # for x in user_average.take(10):
    #     print(x)

    business_average = data.map(lambda (user, business, star, astar): (business, (star, astar, 1))).reduceByKey(lambda (s_x, as_x, c_x), (s_y, as_y, c_y) : (s_x + s_y, as_x + as_y, c_x + c_y)).map(lambda (business, (s, ads, c)) : (business, (float(s)/float(c), float(ads)/float(c))))

    aggregate_data = data.map(lambda (u, b, s, ads) : (u, (b, s, ads))).join(user_average).map(lambda (u, ((b, s, ads), (us, uads))) : (b, ((u, us, uads), (s, ads)))).join(business_average).map(lambda (b, (((u, us, uads), (s, ads)), (bs, bads))) : ((u, us, uads), (b, bs, bads), (s, ads)))

    diff_data = aggregate_data.map(lambda ((u, us, uads), (b, bs, bads), (s, ads)) : ((us - bs, uads - bads), (s - bs, ads - bads)))
    # diff_data = aggregate_data.map(lambda ((u, us, uads), (b, bs, bads), (s, ads)) : ((s - us, ads - uads), (s - bs, ads - bads)))
    diff_original = diff_data.map(lambda ((ds_one, dads_one), (ds_two, dads_two)) : (ds_one, ds_two))
    diff_adjusted = diff_data.map(lambda ((ds_one, dads_one), (ds_two, dads_two)) : (dads_one, dads_two))

    diff_original_plot = diff_original.collect()
    diff_adjusted_plot = diff_adjusted.collect()

    plt.subplot(1, 2, 1)
    plt.plot(map(lambda (ds_one, ds_two): ds_two, diff_original_plot), map(lambda (ds_one, ds_two): ds_one, diff_original_plot),'o')
    plt.title('Original Stars')
    plt.xlabel('star - business_average')
    plt.ylabel('user_average - business_average')
    # plt.ylabel('star - user_average')

    plt.subplot(1, 2, 2)
    plt.plot(map(lambda (dads_one, dads_two): dads_two, diff_adjusted_plot), map(lambda (dads_one, dads_two): dads_one, diff_adjusted_plot),'o')
    plt.title('Adjusted Stars')
    plt.xlabel('star - business_average')
    plt.ylabel('user_average - business_average')
    # plt.ylabel('star - user_average')

    plt.show()

    # End time
    end_time = time.time()
    print("Total execution time: " + str(end_time - start_time))
