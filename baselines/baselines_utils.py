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
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, roc_auc_score, plot_roc_curve, roc_curve, average_precision_score
import operator
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import label_binarize

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
		dataset[target] = np.where(dataset[target] < 0, 0, 1) 
		
		dataset['PP_'+target] = stats.zscore(np.asarray(dataset['PP_'+target]))
		dataset['PP_'+target] = np.where(dataset['PP_'+target] < 0, 0, 1) 

		# dataset['Match_'+target] = np.where((dataset[target] == 1) & (dataset['PP_'+target]) == 1, 1, 0) 
		dataset['Match_'+target] = dataset[target] & dataset['PP_'+target]

		# dataset[target] = label_binarize(dataset[target], classes=[1, 0])
		# dataset['PP_'+target] = label_binarize(dataset['PP_'+target], classes=[1, 0])
		# dataset['Match_'+target] = label_binarize(dataset['Match_'+target], classes=[1, 0])

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

# def get_roc_curve(classifier, X, y, target_name, trains, tests, x_test, y_test, figure_tag='_none_'):
def get_roc_curve(classifier, X, y, target_name, trains, tests, figure_tag='_none_'):

	# #############################################################################

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	predict_probas = []
	decision_function = []
	conf_matrix_list_of_arrays = []
	predictions = []
	precision = []
	recall = []
	aucs_pr = []
	ap_pr = []
	ar_pr = []

	fig, ax = plt.subplots(figsize=(10,7), dpi=100)
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
											label='Chance', alpha=.8)

	fig_pr, ax_pr = plt.subplots(figsize=(10,7), dpi=100)

	i = 0

	y_real = []
	y_proba = []
	
	for train, test in zip(trains, tests):

		classifier.fit(X[train], y[train])
		predict_probas.append(classifier.predict_proba(X[test]))

		y_score = classifier.decision_function(X[test])
		decision_function.append(y_score)

		# decision_function.append(y[test])
		predict = classifier.predict(X[test])
		predictions.append(predict)

		# print(y[test], predict)
		conf_matrix = confusion_matrix(y[test], predict)
		conf_matrix_list_of_arrays.append(conf_matrix)

		######## PR
		y_real.append(y[test])
		y_proba.append(predict)

		prec, rec, _ = precision_recall_curve(y[test], y_score)

		# precision.append(prec)
		# recall.append(rec)

		AUC = auc(rec, prec)
		AP = np.mean(prec)
		AR = np.mean(rec)

		aucs_pr.append(AUC)
		ap_pr.append(AP)
		ar_pr.append(AR)
		ax_pr.plot(rec, prec,
								lw=1, alpha=0.4,
				 				label='PR fold %d (AUC = %0.2f)' % (i, AUC)
				 				)
		#####################################

		viz = plot_roc_curve(classifier, X[test], y[test],
													name='ROC fold {}'.format(i),
													alpha=0.4, lw=1, ax=ax)

		interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(viz.roc_auc)
		i += 1

	############# PR CURVE ###########################################
	y_real = np.concatenate(y_real)
	y_proba = np.concatenate(y_proba)

	mean_precisio, mean_recall, _ = precision_recall_curve(y_real, y_proba)

	# mean_precisio = np.mean(precision, axis=0, dtype=np.float64)
	# mean_recall = np.mean(recall, axis=0, dtype=np.float64)

	mean_auc_pr = np.mean(aucs_pr)
	mean_ap_pr = np.mean(ap_pr)
	mean_ar_pr = np.mean(ar_pr)

	std_auc_pr = np.mean(aucs_pr)
	std_ap_pr = np.mean(ap_pr)
	std_ar_pr = np.mean(ar_pr)

	ax_pr.plot(mean_recall, mean_precisio, color='b',
			 label=r'Precision-Recall (AUC = %0.2f $\pm$ %0.2f, AP = %0.2f $\pm$ %0.2f, AR = %0.2f $\pm$ %0.2f)' % (mean_auc_pr, std_auc_pr, mean_ap_pr, std_ap_pr, mean_ar_pr, std_ar_pr),
			 lw=2, alpha=.8)
	ax_pr.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], 
								title="PR curves for " + str(target_name) + " target.")
	ax_pr.legend(loc="upper right")
	ax_pr.grid(True)
	ax_pr.set_xlabel("Recall (Positive label: 1)")
	ax_pr.set_ylabel("Precision (Positive label: 1)")
	# plt.show()
	ax_pr.figure.savefig("pr_curve_"+str(figure_tag)+"_"+str(target_name)+".pdf")
	# ax_pr.close()

	############# ROC CURVE ###########################################


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
	ax.grid(True)
	# plt.show()
	ax.figure.savefig("roc_auc_"+str(figure_tag)+"_"+str(target_name)+".pdf")
	plt.close()
	return mean_auc, std_auc, predict_probas, decision_function, np.mean(conf_matrix_list_of_arrays, axis=0), np.std(conf_matrix_list_of_arrays, axis=0), predictions


def majority_vote(predict_probas_v, predict_probas_w, decision_function_v, decision_function_w, tests, y_test, target_name, figure_tag='_none_'):

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	fig, ax = plt.subplots(figsize=(10,7), dpi=100)
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
											label='Chance', alpha=.8)

	decision_function = []
	for des_func_v, des_func_w in zip(decision_function_v, decision_function_w):

		dfavg = np.array((des_func_v,des_func_w))
		decision_function.append(np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[1], axis=0, arr=dfavg))

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
	return mean_auc, std_auc, prob, decision_function

# def get_roc_curve_multimodal(classifier, X_list, y_list, target_name, trains, tests, x_test_list, y_test_list, figure_tag='_none_'):
def get_roc_curve_multimodal(classifier, X_list, y_list, target_name, trains, tests, figure_tag='_none_'):

	# #############################################################################

	X0 = X_list[0]
	X1 = X_list[1]
	y0 = y_list[0]
	y1 = y_list[1]

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)

	predict_probas = []
	decision_function = []
	conf_matrix_list_of_arrays = []
	precision = []
	recall = []
	aucs_pr = []
	ap_pr = []
	ar_pr = []


	fig, ax = plt.subplots(figsize=(10,7), dpi=100)
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
											label='Chance', alpha=.8)
	fig_pr, ax_pr = plt.subplots(figsize=(10,7), dpi=100)

	i = 0

	y_real = []
	y_proba = []

	for train, test in zip(trains, tests):
		
		classifier.fit([X0[train], X1[train]], [y0[train], y1[train]])
		predict_proba = classifier.predict_proba([X0[test], X1[test]])
		predict_probas.append(predict_proba)
		y_score = classifier.decision_function([X0[test], X1[test]])
		decision_function.append(y_score)

		predict = classifier.predict([X0[test], X1[test]])

		conf_matrix = confusion_matrix(y0[test], predict)
		conf_matrix_list_of_arrays.append(conf_matrix)

		######## PR
		y_real.append(y0[test])
		y_proba.append(y_score)

		prec, rec, _ = precision_recall_curve(y0[test], predict_proba[:,1])

		precision.append(prec)
		recall.append(rec)
		# AUC = auc(rec, prec)
		AUC = roc_auc_score(y0[test], predict_proba[:, 1])
		AP = np.mean(prec)
		AR = np.mean(rec)
		aucs_pr.append(AUC)
		ap_pr.append(AP)
		ar_pr.append(AR)
		ax_pr.plot(rec, prec,
								lw=1, alpha=0.4,
				 				label='PR fold %d (AUC = %0.2f)' % (i, AUC)
				 				)
		#####################################

		fpr, tpr, _ = roc_curve(y0[test],y_score)
		roc_auc = auc(fpr, tpr)
		ax.plot(fpr, tpr, 
						label='ROC fold {} (AUC {:.2f})'.format(i,roc_auc), 
						lw=1, 
						alpha=0.4)

		interp_tpr = np.interp(mean_fpr, fpr, tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(roc_auc)
		i += 1

	############# PR CURVE ###########################################
	y_real = np.concatenate(y_real)
	y_proba = np.concatenate(y_proba)

	mean_precisio, mean_recall, _ = precision_recall_curve(y_real, y_proba)
	# mean_precisio = np.mean(aucs_pr, axis=0, dtype=np.float64)
	# mean_recall = np.mean(aucs_pr, axis=0, dtype=np.float64)

	mean_auc_pr = np.mean(aucs_pr)
	mean_ap_pr = np.mean(ap_pr)
	mean_ar_pr = np.mean(ar_pr)

	std_auc_pr = np.mean(aucs_pr)
	std_ap_pr = np.mean(ap_pr)
	std_ar_pr = np.mean(ar_pr)

	ax_pr.plot(mean_recall, mean_precisio, color='b',
			 label=r'Precision-Recall (AUC = %0.2f $\pm$ %0.2f, AP = %0.2f $\pm$ %0.2f, AR = %0.2f $\pm$ %0.2f)' % (mean_auc_pr, std_auc_pr, mean_ap_pr, std_ap_pr, mean_ar_pr, std_ar_pr),
			 lw=2, alpha=.8)
	ax_pr.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], 
								title="PR curves for " + str(target_name) + " target.")
	ax_pr.legend(loc="upper right")
	ax_pr.grid(True)
	# plt.show()
	ax_pr.figure.savefig("pr_curve_"+str(figure_tag)+"_"+str(target_name)+".pdf")
	# ax_pr.close()

	############# ROC CURVE ###########################################

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
	ax.grid(True)
	# plt.show()
	ax.figure.savefig("roc_auc_"+str(figure_tag)+"_"+str(target_name)+".pdf")
	plt.close()
	return mean_auc, std_auc, predict_probas, decision_function, np.mean(conf_matrix_list_of_arrays, axis=0), np.std(conf_matrix_list_of_arrays, axis=0)

class majority_vote_class(BaseEstimator, ClassifierMixin):
	"""
	Ensemble classifier for scikit-learn estimators.

	Parameters
	----------

	clf : `iterable`
	  A list of scikit-learn classifier objects.
	weights : `list` (default: `None`)
	  If `None`, the majority rule voting will be applied to the predicted class labels.
		If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
		will be used to determine the most confident class label.

	"""
	def __init__(self, clfs, weights=None):
		self.clfs = clfs
		self.weights = weights

	def fit(self, X, y):
		"""
		Fit the scikit-learn estimators.

		Parameters
		----------

		X : numpy array, shape = [n_samples, n_features]
			Training data
		y : list or numpy array, shape = [n_samples]
			Class labels

		"""
		i = 0
		for clf in self.clfs:
			clf.fit(X[i], y[i])
			i+=1

	def predict_proba(self, X):

		"""
		Parameters
		----------

		X : numpy array, shape = [n_samples, n_features]

		Returns
		----------

		avg : list or numpy array, shape = [n_samples, n_probabilities]
			Weighted average probability for each class per sample.

		"""
		self.probas_ = []
		i=0
		for clf in self.clfs:
			self.probas_.append(clf.predict_proba(X[i]))
			i+=1

		# avg = np.average(self.probas_, axis=0, weights=self.weights)
		max_val = np.max(self.probas_, axis=0)

		return max_val

	def decision_function(self, X):

		"""
		Parameters
		----------

		X : numpy array, shape = [n_samples, n_features]

		Returns
		----------

		avg : list or numpy array, shape = [n_samples, n_probabilities]
			Weighted average probability for each class per sample.

		"""
		self.decision_ = []
		i=0
		for clf in self.clfs:
			self.decision_.append(clf.decision_function(X[i]))
			i+=1

		avg = np.average(self.decision_, axis=0, weights=self.weights)

		return avg

	def predict(self, X):
		"""
		Parameters
		----------

		X : numpy array, shape = [n_samples, n_features]

		Returns
		----------

		maj : list or numpy array, shape = [n_samples]
			Predicted class labels by majority rule

		"""
		self.classes_ = []
		i=0
		for clf in self.clfs:

			self.classes_.append(np.asarray(clf.predict(X[i])))
			i+=1

		if self.weights:
			avg = self.predict_proba(X)

			maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

		else:
			maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

		return maj

def img_confusion_matrix(cm, cm_std, classes, title = 'Normalized confusion matrix'):

	# cm = cm.astype('float') / cm.sum()
	# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() - cm.min()
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, "{:.2f}+-{:.2f}".format(cm[i, j],cm_std[i, j]),
				ha="center", va="center",
				color="white" if cm[i, j] > cm.max() - thresh/2 else "black")

	fig.tight_layout()
	new_title = title.replace(" ", "_")
	# plt.grid(True)
	plt.savefig("conf_mtx_"+str(new_title)+".pdf")
	plt.close()
	# plt.show()

def draw_cv_pr_curve(classifier, cv, X, y, title='PR Curve'):
	"""
	Draw a Cross Validated PR Curve.
	Keyword Args:
		classifier: Classifier Object
		cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
		X: Feature Pandas DataFrame
		y: Response Pandas Series
		
	Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
	"""
	y_real = []
	y_proba = []

	i = 0
	for train, test in cv.split(X, y):
		probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
		# Compute ROC curve and area the curve
		precision, recall, _ = precision_recall_curve(y.iloc[test], probas_[:, 1])
		
		# Plotting each individual PR Curve
		plt.plot(recall, precision, lw=1, alpha=0.3,
				 label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))
		
		y_real.append(y.iloc[test])
		y_proba.append(probas_[:, 1])

		i += 1
	
	y_real = np.concatenate(y_real)
	y_proba = np.concatenate(y_proba)
	
	precision, recall, _ = precision_recall_curve(y_real, y_proba)

	plt.plot(recall, precision, color='b',
			 label=r'Precision-Recall (AP = %0.2f)' % (average_precision_score(y_real, y_proba)),
			 lw=2, alpha=.8)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.show()