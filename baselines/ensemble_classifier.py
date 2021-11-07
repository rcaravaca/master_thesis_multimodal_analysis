from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import operator

class majority_vote():
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
	def __init__(self, clfs):
		self.clfs = clfs

	def fit(self, X, Y):
		"""
		Fit the scikit-learn estimators.

		Parameters
		----------

		X : numpy array, shape = [n_samples, n_features]
			Training data
		y : list or numpy array, shape = [n_samples]
			Class labels

		"""	
		for (clf,x,y) in zip(self.clfs, X, Y):
			clf.fit(x, y)

	def predict(self, Xs):
		"""
		Parameters
		----------

		X : numpy array, shape = [n_samples, n_features]

		Returns
		----------

		maj : list or numpy array, shape = [n_samples]
			Predicted class labels by majority rule

		"""
		avg = self.predict_proba(Xs)
		maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
		prob = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[1], axis=1, arr=avg)

		return maj, prob

	def predict_proba(self, Xs):

		"""
		Parameters
		----------

		X : numpy array, shape = [n_samples, n_features]

		Returns
		----------
	
		avg : list or numpy array, shape = [n_samples, n_probabilities]
			Weighted average probability for each class per sample.

		"""
		self.probas_ = [clf.predict_proba(X) for (clf,X) in zip(self.clfs, Xs)]
		avg = np.average(self.probas_, axis=0)

		return avg

	def get_roc_curve(self, Xs, y_test):

		classes, prob = self.predict(Xs)
		fpr, tpr, thresholds = roc_curve(y_test, prob)
		auc = roc_auc_score(y_test, prob)

		return fpr, tpr, thresholds, auc


