# inf553-adjusted-ratings
This repository is the project on "Yelp Recommendation Based on Aggregated Ratings from Sentiment and Stars" in class 
INF-553 at University of Southern California (USC). Our work attempts to combine the sentiment scores from the text content as well as their star ratings in order to create a new normalized rating value. This new rating will then be used in a collaborative filtering recommendation system which will be proven to recommend businesses that the previous system on solely stars would not.

## Dataset Extraction
Scripts to pull information from the entire yelp review dataset

## Sentiment Analysis
We ran various sentiment analysis algorithms over the set and compared the results given to select on one. After seleciton, a new script was used to process sentiment in batches.

## Satistical Analysis
We used statistical analysis of the original star ratings and the sentiment scores in order to understand how to normalize the ratings so as to create a new score that is a more accurate representation of the user's feedback about a business.

## New Scores
The new scores are adjusted to the same scale so that the overall user ratings are centered around 3, and follow a normal distribution. This helps to correct for easy vs. harsh raters, reduce user-business bias, and provides a standardized scale by which to compare businesses to each other.

## Clustering
K-Means is planned to use to cluster the users based on the new adjusted stars and the original stars. The expected goal of clustering is to group the users that are similar based on adjusted rating in the same group and leave biased users on other different groups.

## Recommendation Systems
The recommendation system is a user-to-user collaborative filtering model built in Scala. The baseline comparison metric for a recommendation system is root mean square error (RMSE). In order to test similarity between the results from the recommendation system on stars versus adjusted stars, we find the intersection of those user-review combinations that have an RMSE of under 1 and finding the intersection percentage of these in the test set.

## Visualization
We created a scatter plot to visualize and at same time evaluate our new rating systems. This approach is based on observing the difference of the stars from user average and business average values. 

## Authors 
* Justyn Lee - justynle@usc.edu
* David Goodfellow - dgoodfel@usc.edu
* Tu Mai Anh Do - tudo@usc.edu

