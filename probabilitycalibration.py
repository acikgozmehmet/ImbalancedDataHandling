import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


class ProbabilityCalibration:
	"""
	estimator: estimator already fitted using training data  

	"""


	def __init__(self, estimator, method:str='sigmoid', cv:any='prefit', ensemble:bool=False):
	# def __init__(self, estimator, method:str='sigmoid', cv:any=None, ensemble:bool=True):
		self.estimator = estimator
		self.method = method
		self.cv= cv
		self.ensemble= ensemble

		self.calibrator = CalibratedClassifierCV(self.estimator, method=self.method, cv= self.cv, ensemble = self.ensemble)
		self.brier_score_loss = None


	def fit(X_train, y_train):
		self.estimator.fit(X_train, y_train)


	def predict_proba(X_test, y_test):
		probas = self.calibrator.predict_proba(X_test)[:,1]
		self.brier_score_loss = brier_score_loss(y_test, probas)

		return probas


	@static
	def plot_calibration_curve(y_true, probs, bins, strategy):

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=bins, strategy=strategy)
    
    max_val = max(mean_predicted_value)
    
    plt.figure(figsize=(8,10))
    plt.subplot(2, 1, 1)
    plt.plot(mean_predicted_value, fraction_of_positives, label='Random Forests')
    plt.plot(np.linspace(0, max_val, bins), np.linspace(0, max_val, bins),
         linestyle='--', color='red', label='Perfect calibration')
    
    plt.xlabel('Probability Predictions')
    plt.ylabel('Fraction of positive examples')
    plt.title('Calibration Curve')
    plt.legend(loc='upper left')


    plt.subplot(2, 1, 2)
    plt.hist(probs, range=(0, 1), bins=bins, density=True, stacked=True, alpha=0.3)
    plt.xlabel('Probability Predictions')
    plt.ylabel('Fraction of examples')
    plt.title('Density')
    plt.show()