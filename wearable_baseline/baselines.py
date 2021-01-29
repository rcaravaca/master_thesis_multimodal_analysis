import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, roc_auc_score, plot_roc_curve, roc_curve

# def get_parser():
# 	parser = argparse.ArgumentParser(description="video baseline")
# 	parser.add_argument("-t","--target", help="Path to video file.")
	
# 	return parser

def replace_values(dataset, val, targets):
	
	for t in targets:
		print("INFO: Replacing values for target : ",t)
		median = round(dataset[t].median())
		# print("---------------------------------- \nINFO: Removing",val,"in targets: \n",dataset[t].value_counts())
		dataset[t].replace(to_replace=val, value = median, inplace = True)

	return dataset

def binarize_target(dataset,target,plot_dist):

	dataset[target] = stats.zscore(np.asarray(dataset[target]))

	if plot_dist:
		ax = sns.distplot(dataset[target])
		plt.show()

	dataset[target] = np.where(dataset[target] <= 0, False, True) 
	# ax = sns.distplot(binarized)
	# plt.show()

	# unique, counts = np.unique(binarized, return_counts=True)
	# print(dict(zip(unique, counts)))

	return dataset[target]

def grid_search_wrapper(model, X_train, X_test, y_train, y_test, param_grid, scorers, refit_score, splits=10):
	"""
	fits a GridSearchCV classifier using refit_score for optimization
	prints classifier performance metrics
	"""
	skf = StratifiedKFold(n_splits=splits)
	grid_search = GridSearchCV(model, param_grid, scoring=scorers, refit=refit_score,
							cv=skf, return_train_score=True, n_jobs=-1)
	grid_search.fit(X_train, y_train)

	# make the predictions
	y_pred = grid_search.predict(X_test)

	print('Best params for {}'.format(refit_score))
	print(grid_search.best_params_)

	# confusion matrix on the test data.
	print('\nConfusion matrix of model optimized for {} on the test data:'.format(refit_score))
	print(pd.DataFrame(confusion_matrix(y_test, y_pred),
				 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
	return grid_search


def get_roc_curve(classifier, X, y, target_name, splits=6):

	# #############################################################################
	# Classification and ROC analysis
	cv = StratifiedKFold(n_splits=splits)

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	# print(X.shape)

	fig, ax = plt.subplots(figsize=(10,7), dpi=100)
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
											label='Chance', alpha=.8)

	for i , (train, test) in enumerate(cv.split(X, y)):

		classifier.fit(X[train], y[train])
		viz = plot_roc_curve(classifier, X[test], y[test],
													name='ROC fold {}'.format(i),
													alpha=0.4, lw=1, ax=ax)

		interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(viz.roc_auc)

	
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)

	ax.plot(mean_fpr, mean_tpr, color='b', 
								label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
								lw=2, 
								alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	ax.fill_between(mean_fpr, tprs_lower, tprs_upper, 
											color='grey', 
											alpha=.2, 
											label=r'$\pm$ 1 std. dev.')

	ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], 
								title="ROC curves for " + str(target_name) + " target.")

	ax.legend(loc="lower right")
	# plt.show()
	plt.savefig("roc_auc_"+str(target_name)+".png")




def get_roc_curve2(classifier, X, y, splits=10):

	cv = StratifiedKFold(n_splits=10)

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	plt.figure(figsize=(8,8))
	i = 0
	for train, test in cv.split(X, y):
		probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
		# Compute ROC curve and area the curve
		fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
		tprs.append(np.interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)
		plt.plot(fpr, tpr, lw=1, alpha=0.3,
				 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

		i += 1
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
			 label='Chance', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b',
			 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
			 lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
					 label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	plt.xlabel('False Positive Rate',fontsize=18)
	plt.ylabel('True Positive Rate',fontsize=18)
	plt.title('Cross-Validation ROC of SVM',fontsize=18)
	plt.legend(loc="lower right", prop={'size': 15})
	plt.show()