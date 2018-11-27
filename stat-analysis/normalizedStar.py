# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:07:23 2018

@author: justy
"""

from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

def normalizeValue(minVal, maxVal, val):
    rangeVal = maxVal - minVal
    return (val - minVal) / rangeVal
  
def normalizeValueDist(star, tscore):
    return 3 + (star-3)/1.5 + tscore/3

def buckets(user, val):
    if val == 1:
        return (user, (1,0,0,0,0))
    elif val == 2:
        return (user, (0,1,0,0,0))
    elif val == 3:
        return (user, (0,0,1,0,0))
    elif val == 4:
        return (user, (0,0,0,1,0))
    else:
        return (user, (0,0,0,0,1))

def modeNum(buckets):
    count = 0
    if buckets[0] > buckets[1]:
        count += 1
    if (buckets[1] > buckets[0]) & (buckets[1] > buckets[2]):
        count += 1
    if (buckets[2] > buckets[1]) & (buckets[2] > buckets[3]):
        count += 1
    if (buckets[3] > buckets[2]) & (buckets[3] > buckets[4]):
        count += 1
    if buckets[4] > buckets[3]:
        count += 1
    count = 1 if count == 0 else count
    return count

def SampleStDev(val, count):
    if count == 1:
        return 0.0
    else:
        return np.sqrt(val/(count-1))

def TScore(val, mean, std):
    if std == 0.0:
        return 0.0
    else:
        return (val - mean)/std


def getTScore(scoreMap, reviewId):
    if reviewId in scoreMap:
        return scoreMap[reviewId]
    else:
        return 0.0

def getAdjStars(sentStar, modes, reviews, mean, tscore, lowT, highT):
    if modes == 1:
        if reviews == 1:
            return mean
        else:
            return tscore + 3
    elif modes == 2:
        if sentStar < 3:
            return lowT / 2 + 2.5
        else:
            return highT / 2 + 3
    else:
        return tscore + 3

def getNormalizeStars(dfSent):
  t0 = time.clock()
  rddStart = dfSent.rdd
  rddSent = rddStart.map(lambda row: (row['user_id'], row['stars'], row['sentiment-google'], row['review_id']))
  
  ''' ***** Get expanded star ratings using sentiment, summary stats ***** '''
  print('***** Get expanded star ratings using sentiment, summary stats *****')
  
  # (star: mean sentiment)
  meanMap = rddSent.map(lambda x: (x[1], (x[2], 1))) \
      .reduceByKey(lambda sumCnt1, sumCnt2: (sumCnt1[0]+sumCnt2[0], sumCnt1[1]+sumCnt2[1])) \
      .map(lambda sumCnt: (sumCnt[0], sumCnt[1][0]/sumCnt[1][1])).collectAsMap() 
  
  # (star: std dev sentiment)
  stdMap = rddSent.map(lambda x: (x[1], ((x[2] - meanMap[x[1]])**2,1) ) ) \
      .reduceByKey(lambda sumCnt1, sumCnt2: (sumCnt1[0]+sumCnt2[0], sumCnt1[1]+sumCnt2[1])) \
      .map(lambda sumCnt: (sumCnt[0], SampleStDev(sumCnt[1][0],sumCnt[1][1]) )).collectAsMap()
  
  # (review_id: t-score sentiment)
  tscoreStar = rddSent.map(lambda x: (x[3], TScore(x[2],meanMap[x[1]],stdMap[x[1]]) ) ).collectAsMap()
  
  # (user_id, expanded star using sentiment, review_id, stars)
  rddSentStars = rddSent.map(lambda row: (row[0], normalizeValueDist(row[1], tscoreStar[row[3]]), \
                                          row[3], row[1]))
  
  # review count, mean by user_id
  countsMap = rddSentStars.map(lambda x: (x[0], (x[1], 1))) \
      .reduceByKey(lambda cnt1, cnt2: (cnt1[0]+cnt2[0], cnt1[1]+cnt2[1]) ) \
      .map(lambda cntAvg: (cntAvg[0], (cntAvg[1][1], cntAvg[1][0]/cntAvg[1][1])) ).collectAsMap()
      
  elapsed = time.clock() - t0
  print("Execution time: %f\n" % elapsed)
  
  ''' ***** Get number of modes per user ***** '''
  print('***** Get number of modes per user *****')
  t0 = time.clock()
  
  # (user_id, bucket counts tuple)
  rddBuckets = rddSent.map(lambda row: buckets(row[0], row[1])) \
      .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3], x[4]+y[4]))
  
  # (user_id: number of modes)
  modesMap = rddBuckets.map(lambda x: (x[0], modeNum(x[1]))).collectAsMap()
  elapsed = time.clock() - t0
  print("Execution time: %f\n" % elapsed)
  
  ''' ***** Get T-scores for full dist, low dist, and high dist ***** '''
  print('***** Get T-scores for full dist, low dist, and high dist *****')
  t0 = time.clock()
  
  rddFull = rddSentStars.flatMap(lambda x: [(x[0], x[1], x[2])] )
  rddHigh = rddSentStars.flatMap(lambda x: [(x[0], x[1], x[2])] if x[1] >= 3 else [] )
  rddLow = rddSentStars.flatMap(lambda x: [(x[0], x[1], x[2])] if x[1] < 3 else [] )
  
  tscoreList = [rddFull, rddLow, rddHigh]
  
  
  lowHighSum = []
  for rdd in tscoreList:
  
      # (user_id: mean)
      meanMap = rdd.map(lambda x: (x[0], (x[1], 1))) \
          .reduceByKey(lambda sumCnt1, sumCnt2: (sumCnt1[0]+sumCnt2[0], sumCnt1[1]+sumCnt2[1])) \
          .map(lambda sumCnt: (sumCnt[0], sumCnt[1][0]/sumCnt[1][1])).collectAsMap() 
  
      # (user_id, std dev)
      stdMap = rdd.map(lambda x: (x[0], ((x[1] - meanMap[x[0]])**2,1) ) ) \
          .reduceByKey(lambda sumCnt1, sumCnt2: (sumCnt1[0]+sumCnt2[0], sumCnt1[1]+sumCnt2[1])) \
          .map(lambda sumCnt: (sumCnt[0], SampleStDev(sumCnt[1][0],sumCnt[1][1]) )).collectAsMap()
      
      # (review_id, t-score)
      tscoreRdd = rdd.map(lambda x: (x[2], TScore(x[1],meanMap[x[0]],stdMap[x[0]]) ) ).collectAsMap()
      lowHighSum.append(tscoreRdd)
      
  elapsed = time.clock() - t0
  print("Execution time: %f\n" % elapsed)
      
  ''' ***** Get final normalized star rating ***** '''
  print('***** Get final normalized star rating *****')
  t0 = time.clock()
  
  fullMap = lowHighSum[0]
  lowMap  = lowHighSum[1]
  highMap = lowHighSum[2]
  
  # (review_id, sentiment stars, # of modes, # of reviews, user mean, full t-score, low t-score, high t-score, \
  #              user_id, stars)
  rddAll = rddSentStars.map(lambda x: (x[2], x[1], modesMap[x[0]], countsMap[x[0]][0], \
                                       countsMap[x[0]][1], getTScore(fullMap, x[2]), \
                                       getTScore(lowMap, x[2]), getTScore(highMap, x[2]), \
                                       x[0], x[3]) )
  
  # map of (review_id : business_id)
  businessMap = rddStart.map(lambda row: (row['review_id'], row['business_id'])).collectAsMap()
  
  def getBusiness(bizMap, review_id):
      return (review_id, bizMap[review_id])
  
  rddAdjStars = rddAll.map(lambda x: (getAdjStars(x[1], x[2], x[3], x[4], x[5], x[6], x[7]), \
                            businessMap[x[0]], x[8], x[9])).collect()
  
  elapsed = time.clock() - t0
  print("Execution time: %f\n" % elapsed)                         
  return rddAdjStars

def plotData(adjStarsList, numStr):
  QAll = pd.DataFrame(adjStarsList, columns=['Adj_Stars', 'business_id', 'user_id', 'stars'])

  plt.hist(QAll['stars'], bins=[0.5,1.5,2.5,3.5,4.5,5.5], color='orange')
  plt.title("Sentiment Stars Distribution" + numStr)
  plt.xlabel("Star Rating")
  plt.ylabel("Review Count")
  plt.show()
  
  plt.hist(QAll['Adj_Stars'], bins=[0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75], color='green')
  plt.title("Normalized Stars Distribution" + numStr)
  plt.xlabel("Normalized Star Rating")
  plt.ylabel("Review Count")
  plt.show()
  
  std1 = len(QAll[(QAll['Adj_Stars'] < 4) & (QAll['Adj_Stars'] > 2)])/len(QAll)
  print("Within 1 StD: %f" % std1)
  std2 = len(QAll[(QAll['Adj_Stars'] < 5) & (QAll['Adj_Stars'] > 1)])/len(QAll)
  print("Within 2 StD: %f" % std2)
  return QAll

spark = SparkSession \
    .builder \
    .appName("Normalize") \
    .config("spark.local.dir","C:\\Users\\justy\\Documents\\USC\\Fall 2018\\INF 553\\Project") \
    .getOrCreate()

for i in range(0,190,10):
  numStr = str(i) + str(i+10)
  filePath = "C:/Users/justy/Documents/GitHub/inf553-adjusted-ratings/" + numStr + "google-results.json"
  
  # normalized stars (first 1000 reviews)
  dfSent = spark.read.json(filePath, multiLine=True)
  adjStarsList = getNormalizeStars(dfSent)
  QAll = plotData(adjStarsList, numStr)
  
  # Output results to json file
  outputFile = "user_normalizedSentStars_" + numStr + ".json"
  QAll.to_json(outputFile, orient='records')
















