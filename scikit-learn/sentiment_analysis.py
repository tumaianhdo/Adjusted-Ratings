"""Build a sentiment analysis model with scikit-learn

"""
# Author: Tu Mai Anh Do <tudo@usc.edu>

import json
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# the data path must be passed as first argument
	data_path = sys.argv[1]
	with open(data_path) as data_file:
		data = json.load(data_file)

		 # split the dataset in training and test set:
		train_review = [data[i]['text'] for i in range(0, 750)]
		train_star = [data[i]['stars'] for i in range(0, 750)]
		test_review = [data[i]['text'] for i in range(750, 1000)]
		test_star = [data[i]['stars'] for i in range(750, 1000)]
		# print(data)
	
	print("# samples: %d" % len(data))
	print("# train samples: %d" % len(train_review))
	print("# test samples: %d" % len(test_review))


	# Build a vectorizer / classifier pipeline that filters out tokens that are too rare or too frequent
	pipeline = Pipeline([
		('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
		('clf', LinearSVC()),
	])

	# Build a grid search to find out whether unigrams or bigrams are more useful.
	parameters = {
		'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
		'clf__C': [1, 10, 100, 1000],
	}

	# Fit the pipeline on the training set using grid search for the parameters
	grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
	grid_search.fit(train_review, train_star)

	# Print the mean and std for each candidate along with the parameter
	# settings for all the candidates explored by grid search.
	n_candidates = len(grid_search.cv_results_['params'])
	for i in range(n_candidates):
		print(i, 'params - %s; mean - %0.2f; std - %0.2f'
				 % (grid_search.cv_results_['params'][i],
					grid_search.cv_results_['mean_test_score'][i],
					grid_search.cv_results_['std_test_score'][i]))

	print("Best score: %0.3f" % grid_search.best_score_)
	print("Best parameters set:")
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))

	# Predict the outcome on the testing set and store it in a variable
	predicted_star = grid_search.predict(test_review)

	# Print the classification report
	print("Classification Report:")
	# print(metrics.classification_report(test_star, predicted_star, target_names=dataset.target_names))
	print(metrics.classification_report(test_star, predicted_star))

	# Print plot the confusion matrix
	print("Confusion Matrix:")
	cm = metrics.confusion_matrix(test_star, predicted_star)
	print(cm)

	# Plot the confusion matrix
	# plt.matshow(cm)
	# plt.show()


