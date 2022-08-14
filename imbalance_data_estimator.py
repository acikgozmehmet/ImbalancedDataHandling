from imblearn import under_sampling, over_sampling, combine, ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import pandas as pd

class ImbalancedClass:
	"""
	It has been tested against binary classification.

	Required library: Imbalanced-learn library
					  https://imbalanced-learn.org/stable/references/index.html

	Please find below the types and methods 

	UnderSample :	
		* Fixed methods: Reduces the majority class to the same number of obs in minority class by selecting from the majority class
			- Random: Randomly delete examples in the majority class
					  It works with both continous and categorical features					  
			
			- NearMiss: Retains obs in the boundary. Class to perform under-sampling based on NearMiss methods. Usually good for text data.

			- InstanceHardnessThreshold: Remove the noisy obs. 
			                             The choices for estimator are the following: None is default which is RamdomForestClassifier
			                             'knn', 'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting' and 'linear-svm'.


		* Cleaning  methods: Clean the majority class based on some criteria by deleting from the majority class
			- CondensedNearestNeighbour (CNN):  Extracts/selects obs at the boundary between/among the classes. It focuses on harder cases to classify
			                                    It introduces noise, though. Undersamples based on the condensed nearest neighbour method.
			                                    It only works with continous features and encoded categorical features. 
			                                    It may be computationally expensive.

			- ClusterCentroids: Undersample by generating centroids based on clustering methods.  By default, it will be a default KMeans estimator.

			- EditedNearestNeighbours : Removes samples from the majority class that are closest to the boundary with the other classes
										Remove the noisy obs. Mostly at the boundary of the classes. Undersample based on the edited nearest neighbour method. 
										This method will clean the database by removing samples close to the decision boundary.

			- RepeatedEditedNearestNeighbours: Remove the noisy obs. Undersample based on the repeated edited nearest neighbour method. 
											   This method will repeat several time the ENN algorithm.

			- AllKNN: Remove the noisy obs.This method will apply ENN several times and will vary the number of nearest neighbours. It removes the hard cases.

			- TomekLinks: Remove the noisy obs. If 2 samples are nearest neighbors, and from a different class, they are Tomek links. It removes the Tomek links 
						  from the majority class by using 1-kNN under the hood. It removes the noise, but it also misclassify the hard cases. 

			- OneSidedSelection: Selects the samples at the boundary of the classes which are hard instances, then removes the Tomek Links (noise)

			- NeighbourhoodCleaningRule: It extends to EditedNearestNeighbours. Remove the noisy obs in the decision boundary

	OverSample:
		Sample extraction methods:		
			- Random: Randomly duplicate examples in the minority class. Shrinkage factor is used to define the dispersion of the extracted samples. 
				      It works with both continous and categorical features	

		Sample generation methods:
			- SMOTE:  Synthetic Minority Over-sampling Technique. Creates samples from minority class by interpolation with KNN. 
			
			- SMOTENC: Synthetic Minority Over-sampling Technique for Nominal and Continuous.
					   Unlike SMOTE, SMOTE-NC for dataset containing numerical and categorical features. 
			           However, it is not designed to work with only categorical features. 
			           categorical_features is the array of indices specifying the categorical features

			- SMOTEN: Synthetic Minority Over-sampling Technique for Nominal. It expects that the data to resample are only made of categorical features.

			- ADASYN:  Adaptive Synthetic (ADASYN) algorithm. More synthetic data is generated from samples that are harder to classify. It uses all samples to train KNN
			           Generates different number of samples depending on an estimate of the local distribution of the class to be oversampled

			- BorderlineSMOTE: It generates synthetic data only from minority class closer to the decision boundary.

			- KMeansSMOTE: It generates synthetic data in intra-class clusttors. Implementation of this method is not very straight-forward. A lot of hyperparameter to tune.

			- SVMSMOTE: Generates synthetic data by inter or extrapolating to neighbours from the minority. If most neughbors are from minority class it uses extrapolation
						If most neighbours are from majority class it uses interpolation.

	Combine: 
			- SMOTEENN: Combination of SMOTE and ENN. Over-sampling using SMOTE and cleaning using ENN. It uses the SMOTE to generate new synthetic data from minority class 
			            while using ENN to removes samples from the majority class that are closest to the boundary with the other classes

			- SMOTETomek: Combination of SMOTE and Tomek links. Over-sampling using SMOTE and cleaning using Tomek links.It uses the SMOTE to generate new synthetic data from minority class
						  while removing the Tomek links from the majority class by using 1-kNN.

	Ensemble:
			- EasyEnsembleClassifier: Creates balanced samples of the training dataset by selecting all examples from the minority class and a subset from the majority class.
			                          Rather than using pruned decision trees, boosted decision trees are used on each subset, specifically the AdaBoost algorithm.

			- RUSBoostClassifier: Random under-sampling integrated in the learning of AdaBoost. 
			                      During learning, the problem of class balancing is alleviated by random under-sampling the sample at each iteration of the boosting algorithm.


			- BalancedBaggingClassifier: A Bagging classifier with additional balancing.
										 This implementation of Bagging is similar to the scikit-learn implementation. It includes an additional step to balance the training set at fit time using a given sampler


			- BalancedRandomForestClassifier: A balanced random forest classifier.
											  A balanced random forest randomly under-samples each boostrap sample to balance it.

	Cost Sensitive Learning:
			- class_weight or sample_weight: Class_weight, it is initilized in the classifier or estimated in hyperparameter tuning. 
			                                 Sample_weight is used with fit method. 

			- MetaCost: Method for making an arbitrary classifier cost-sensitive by wrapping a cost-minimizing procedure around it.
			            It relabels the target to the class that minimizes the condinol risk

																M
										               R(i|x) = Sum [P(j|x) C(i,j) ]
																j=1

    -----
	fit_resample(): Resample the dataset.
	get_resampled_dataframe(): returns the resampled X and y as single dataframe 

	"""

	def __init__(self, dataframe: pd.DataFrame, target: str, type:str, method:str, sampling_strategy:str='auto', random_state:int=None, replacement:bool=False,
					   shrinkage:float or dict=None, k_neighbors:int=5, categorical_features:list=None, n_neighbors:int or object=None, m_neighbors:int or object=10,
					   n_estimators:int=10):

		self.dataframe = dataframe.copy(deep=True)
		self.target = target
		self.type = type
		self.method = method
		self.sampling_strategy = sampling_strategy
		self.random_state = random_state
		self.replacement = replacement
		self.shrinkage = shrinkage
		self.k_neighbors = k_neighbors
		self.categorical_features = categorical_features
		self.n_neighbors = n_neighbors
		self.m_neighbors = m_neighbors
		self.n_estimators = n_estimators


		self.X = self.dataframe.drop([self.target], axis =1)
		self.y = self.dataframe[self.target]

		self.sampler = None


	def _UnderSample(self):
		if self.method =="Random":
			self.sampler = under_sampling.RandomUnderSampler(sampling_strategy= self.sampling_strategy,
											 				 random_state = self.random_state,
											 				 replacement = self.replacement)

		elif self.method =="NearMiss":
			self.sampler = under_sampling.NearMiss(sampling_strategy= self.sampling_strategy,
											       version=1,
											       n_neighbors=3, 
											       n_neighbors_ver3=3)

		elif self.method =="InstanceHardnessThreshold":
			self.sampler = under_sampling.InstanceHardnessThreshold(estimator = None, 
													                sampling_strategy= self.sampling_strategy,
													                random_state = self.random_state,
													                cv=5)

		elif self.method == "CondensedNearestNeighbour":
			self.sampler = under_sampling.CondensedNearestNeighbour(sampling_strategy= self.sampling_strategy,
											                        random_state = self.random_state,
											                        n_neighbors = None, 
											                        n_seeds_S=1)

		elif self.method == "ClusterCentroids":
			self.sampler = under_sampling.ClusterCentroids(sampling_strategy= self.sampling_strategy,
											               random_state = self.random_state,
											               estimator=None,
										                   voting='auto')

		elif self.method == "EditedNearestNeighbours":
			self.sampler = under_sampling.EditedNearestNeighbours(sampling_strategy= self.sampling_strategy,
											                      n_neighbors = 3,
											                      kind_sel='all')	

		elif self.method == "RepeatedEditedNearestNeighbours":
			self.sampler = under_sampling.RepeatedEditedNearestNeighbours(sampling_strategy= self.sampling_strategy,
																		  n_neighbors = 3,
																		  max_iter=100,
																		  kind_sel='all')	

		elif self.method == "AllKNN":
			self.sampler = under_sampling.AllKNN(sampling_strategy= self.sampling_strategy,
												 n_neighbors = 3,
												 kind_sel='all',
												 allow_minority=False)			
		elif self.method == "TomekLinks":
			self.sampler = under_sampling.TomekLinks(sampling_strategy= self.sampling_strategy)			

		elif self.method == "OneSidedSelection":
			self.sampler = under_sampling.OneSidedSelection(sampling_strategy= self.sampling_strategy,
											 				random_state = self.random_state,
											 				n_neighbors = None,
											 				n_seeds_S=1)	

		elif self.method == "NeighbourhoodCleaningRule":
			self.sampler = under_sampling.NeighbourhoodCleaningRule(sampling_strategy= self.sampling_strategy,
													 				n_neighbors=3,
													 				kind_sel='all', 
													 				threshold_cleaning=0.5)	
		else:
			raise Exception("Method not recognized in UnderSampling")


	def _OverSample(self):
		if self.method =="Random":
			"""
			It is resampling with replacement from the minority class
			"""
			self.sampler = over_sampling.RandomOverSampler(sampling_strategy= self.sampling_strategy,
											 random_state = self.random_state,
											 shrinkage = None )

		elif self.method == 'SMOTE':
			self.sampler =  over_sampling.SMOTE(sampling_strategy=self.sampling_strategy, 
												random_state=self.random_state,
												k_neighbors=self.k_neighbors)


		elif self.method == 'SMOTENC':
			self.sampler =  over_sampling.SMOTENC(categorical_features=self.categorical_features,
												  sampling_strategy=self.sampling_strategy, 
												  random_state=self.random_state,
												  k_neighbors=self.k_neighbors)


		elif self.method == 'SMOTEN':
			self.sampler = over_sampling.SMOTEN(sampling_strategy=self.sampling_strategy,
												random_state=self.random_state,
												k_neighbors=self.k_neighbors)

		elif self.method == 'ADASYN':
			self.sampler = over_sampling.ADASYN(sampling_strategy= self.sampling_strategy,
												random_state=self.random_state,
												n_neighbors= self.n_neighbors)

		elif self.method == 'BorderlineSMOTE':
			self.sampler = over_sampling.BorderlineSMOTE(sampling_strategy=self.sampling_strategy,
			                                             random_state=self.random_state,
			                                             k_neighbors=self.k_neighbors, 
			                                             m_neighbors=self.m_neighbors, 
			                                             kind='borderline-1')

		elif self.method == 'KMeansSMOTE':
			self.sampler = over_sampling.KMeansSMOTE(sampling_strategy=self.sampling_strategy,
			                                         random_state=self.random_state,
			                                         k_neighbors=self.k_neighbors, 
			                                         kmeans_estimator=KMeans(n_clusters=self.y.nunique(), random_state=self.random_state), 
			                                         cluster_balance_threshold=0.1, 
			                                         density_exponent='auto')

		elif self.method == 'SVMSMOTE':
			self.sampler = over_sampling.SVMSMOTE(sampling_strategy=self.sampling_strategy,
			                                      random_state=self.random_state,
			                                      k_neighbors=self.k_neighbors, 
			                                      m_neighbors=self.m_neighbors,
			                                      svm_estimator=None, 
			                                      out_step=0.5)
		else:
			raise Exception("Method not recognized in OverSampling")


	def _Combine(self):
		if self.method == 'SMOTEENN':
			self.sampler = combine.SMOTEENN(sampling_strategy=self.sampling_strategy, 
											random_state=self.random_state,
											smote=over_sampling.SMOTE(sampling_strategy=self.sampling_strategy, 
																	   random_state=self.random_state,
																	   k_neighbors=self.k_neighbors),
											enn=under_sampling.EditedNearestNeighbours(sampling_strategy= self.sampling_strategy,
											 										   n_neighbors = 3,
											 										   kind_sel='all'))

		elif self.method == 'SMOTETomek':
			self.sampler = combine.SMOTETomek(sampling_strategy=self.sampling_strategy, 
											  random_state=self.random_state,
											  smote=over_sampling.SMOTE(sampling_strategy=self.sampling_strategy, 
											  						     random_state=self.random_state,
																	     k_neighbors=self.k_neighbors),
											  tomek=under_sampling.TomekLinks(sampling_strategy= self.sampling_strategy))	
		else: 
			raise Exception("Method not recognized in Combine")

	def _Ensemble(self):
		if self.method == "EasyEnsembleClassifier":
			self.sampler = ensemble.EasyEnsembleClassifier(n_estimators=10, 
														   base_estimator=None, 
														   warm_start=False, 
														   sampling_strategy=self.sampling_strategy, 
														   replacement=False, 
														   n_jobs=None, 
														   random_state=self.random_state, 
														   verbose=0)
		elif self.method == "RUSBoostClassifier":
			self.sampler = ensemble.RUSBoostClassifier(base_estimator=None, 
													   n_estimators=50, 
													   learning_rate=1.0, 
													   algorithm='SAMME.R', 
													   sampling_strategy='auto', 
													   replacement=False, 
													   random_state=None)

		elif self.method == "BalancedBaggingClassifier":
			self.sampler = ensemble.BalancedBaggingClassifier(base_estimator=None, 
				                                              n_estimators=10, 
				                                              max_samples=1.0, 
				                                              max_features=1.0, 
				                                              bootstrap=True, 
				                                              bootstrap_features=False, 
				                                              oob_score=False, 
				                                              warm_start=False,
				                                              sampling_strategy='auto',
				                                              replacement=False, 
				                                              n_jobs=None, 
				                                              random_state=None, 
				                                              verbose=0, 
				                                              sampler=None)

		elif self.method == "BalancedRandomForestClassifier":
			self.sampler = ensemble.BalancedRandomForestClassifier( n_estimators=100, 
																	criterion='gini',
																	max_depth=None, 
																	min_samples_split=2, 
																	min_samples_leaf=1, 
																	min_weight_fraction_leaf=0.0, 
																	max_features='auto', 
																	max_leaf_nodes=None, 
																	min_impurity_decrease=0.0, 
																	bootstrap=True, 
																	oob_score=False, 
																	sampling_strategy='auto', 
																	replacement=False, 
																	n_jobs=None, 
																	random_state=None, 
																	verbose=0, 
																	warm_start=False, 
																	class_weight=None, 
																	ccp_alpha=0.0, 
																	max_samples=None)
		else:
			raise Exception("Method not recognized in Ensemble")


		
		# X_balanced, y_balanced = self.sampler.fit_resample(X,y)



	def fit_resample(self):
		if self.type == "UnderSampling":
			self._UnderSample()
		elif self.type == "OverSampling":
			self._OverSample()
		elif self.type == "Combine" :
			self._Combine()
		elif self.type == "Ensemble":
			self._Ensemble()
		else:
			raise Exception("Type not recognized!")
			

		return self.sampler.fit_resample(self.X, self.y)


	def get_resampled_dataframe(self):
		X_resampled, y_resampled = self.fit_resample()
		X_resampled[self.target] = y_resampled
		return X_resampled