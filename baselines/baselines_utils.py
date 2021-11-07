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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, roc_auc_score, plot_roc_curve, roc_curve
import operator


def get_parser():
	parser = argparse.ArgumentParser(description="")
	
	parser.add_argument("-c","--capturar", help="Capturar una imagen. Al precionar la barra espaciadora o la tecla de enter, se debe captura una imagen y se da opcion de recortarla.", action='store_true')


def replace_values(dataset, val, targets):
	
	for t in targets:
		print("-I- Replacing values " + str(val) +" in target : ",t)
		median = round(dataset[t].median())
		# print("---------------------------------- \n-I- Removing",val,"in targets: \n",dataset[t].value_counts())
		dataset[t].replace(to_replace=val, value = median, inplace = True)

	return dataset

def binarize_target(dataset,targets,plot_dist):

	first_four_target = targets[0:4]

	for target in first_four_target:

		if plot_dist:
			ax = sns.distplot(dataset[target])
			plt.show()

		dataset[target] = stats.zscore(np.asarray(dataset[target]))
		dataset[target] = np.where(dataset[target] <= 0, False, True) 
		
		dataset['PP_'+target] = stats.zscore(np.asarray(dataset['PP_'+target]))
		dataset['PP_'+target] = np.where(dataset['PP_'+target] <= 0, False, True) 

		dataset['Match_'+target] = np.where((dataset[target] == True) & (dataset['PP_'+target]) == True, True, False) 

		targets.append('Match_'+target)

	return dataset[targets]
	#, dataset['PP_'+target], dataset['Match_'+target] 

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


def process_dataset(dataset, target_names):

	targets = list(target_names.values())
	
	dataset = dataset.rename(columns=target_names)

	features = dataset.columns.values[1:-12]

	## replacing outlier values in targets
	for target in targets:
		dataset = replace_values(dataset, 999, [target])

	target_col = binarize_target(dataset,targets,False)
	target_col = target_col.drop(['PP_SeeAgain', 'PP_Friendly', 'PP_Sexual','PP_Romantic'], axis=1)

	# Z-Score Normalizing for all features
	for feature in features:
		dataset[feature] = stats.zscore(np.asarray(dataset[feature]))

	dataset = dataset.drop(['date', 'M_1', 'M_2', 'SeeAgain', 'Friendly', 'Sexual', 'Romantic', 'PP_M_1', 'PP_M_2', 'PP_SeeAgain', 'PP_Friendly', 'PP_Sexual', 'PP_Romantic', 'Match_SeeAgain', 'Match_Friendly', 'Match_Sexual', 'Match_Romantic'], axis=1)
	
	return dataset, target_col



def get_roc_curve(classifier, X, y, target_name, trains, tests, figure_tag='_none_'):

	# #############################################################################

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	predict_probas = []

	fig, ax = plt.subplots(figsize=(10,7), dpi=100)
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
											label='Chance', alpha=.8)

	i = 0
	for train, test in zip(trains, tests):
		
		classifier.fit(X[train], y[train])
		predict_probas.append(classifier.predict_proba(X[test]))

		viz = plot_roc_curve(classifier, X[test], y[test],
													name='ROC fold {}'.format(i),
													alpha=0.4, lw=1, ax=ax)


		interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(viz.roc_auc)
		i += 1


	mean_tpr = np.mean(tprs, axis=0, dtype=np.float64)
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
	plt.savefig("roc_auc_"+str(figure_tag)+"_"+str(target_name)+".png")
	plt.close()
	return mean_auc, std_auc, predict_probas


def majority_vote(predict_probas_v, predict_probas_w, tests, y_test, target_name, figure_tag='_none_'):

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	fig, ax = plt.subplots(figsize=(10,7), dpi=100)
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
											label='Chance', alpha=.8)

	i = 0
	for pred_prob_v,pred_prob_w,test in zip(predict_probas_v, predict_probas_w, tests):

		avg = np.average((pred_prob_v,pred_prob_w), axis=0)
		
		classes = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
		prob = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[1], axis=1, arr=avg)
		# print("-----\n",pred_prob_v[:1,:],pred_prob_w[:1,:],avg[:1,:],prob[:1],classes[:1],y_test[test][:1])
		fpr, tpr, thresholds = roc_curve(y_test[test], prob)
		auc_ = roc_auc_score(y_test[test], prob)

		ax.plot(fpr, tpr, 
								label='ROC fold {} (AUC {:.2f})'.format(i,auc_), 
								lw=1, 
								alpha=0.4)


		interp_tpr = np.interp(mean_fpr, fpr, tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(auc_)
		i += 1

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)

	ax.plot(mean_fpr, mean_tpr, color='b', 
								label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
								lw=2, 
								alpha=.8)
	
	# print(' Mean ROC (AUC = %0.2f +- %0.2f)' % (mean_auc, std_auc))

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
	plt.savefig("roc_auc_"+str(figure_tag)+"_"+str(target_name)+".png")
	plt.close()
	return mean_auc, std_auc


def get_roc_curve_multimodal(classifier, XX, YY, target_name, trains, tests, figure_tag='_none_'):

	# #############################################################################
	# Classification and ROC analysis
	# cv = StratifiedKFold(n_splits=splits)

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	fig, ax = plt.subplots(figsize=(10,7), dpi=100)
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
											label='Chance', alpha=.8)

	X0 = XX[0]
	Y0 = YY[0]

	X1 = XX[1]
	Y1 = YY[1]

	i = 0
	for train, test in zip(trains, tests):

		classifier.fit([X0[train], X1[train]], [Y0[train], Y1[train]])

		fpr, tpr, thresholds, auc_ = classifier.get_roc_curve([X0[test], X1[test]], Y0[test])
		ax.plot(fpr, tpr, 
								label='ROC fold {} (AUC {:.2f})'.format(i,auc_), 
								lw=1, 
								alpha=0.4)


		interp_tpr = np.interp(mean_fpr, fpr, tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(auc_)
		i += 1

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)

	ax.plot(mean_fpr, mean_tpr, color='b', 
								label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), 
								lw=2, 
								alpha=.8)
	
	# print(' Mean ROC (AUC = %0.2f +- %0.2f)' % (mean_auc, std_auc))

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
	return mean_auc, std_auc



