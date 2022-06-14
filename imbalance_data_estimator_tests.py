import os
import unittest
from imbalance_data_estimator import ImbalancedClass
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import EasyEnsembleClassifier

class TestImbalancedClassdClass(unittest.TestCase):

	filename = r'C:\mehmet\imbalanced_data\dataset\wrangling_train_dataframe.csv'
	target = 'ProdTaken'

	def _get_target_distribution(self, dataframe: pd.DataFrame):
		counts = dataframe[TestImbalancedClassdClass.target].value_counts().to_list()
		minority_class_size = np.min(counts)
		majority_class_size = np.max(counts)
		return (majority_class_size, minority_class_size)


	def test_RandomUnderSampler(self):
		print("\nRandomUnderSampler test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'TypeofContact_imputed','ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		random_under_sampler = ImbalancedClass(dataframe=actual_df,
										 target=TestImbalancedClassdClass.target, 
										 type='UnderSampling', 
										 method='Random',
										 sampling_strategy='auto',
										 random_state=42)


		expected_df = random_under_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")

		self.assertEqual(expected_df_minority_class_size, actual_df_minority_class_size)
		self.assertEqual(expected_df_majority_class_size, expected_df_minority_class_size)



	def test_RandomOverSampler(self):
		print("\nRandomOverSampler test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'TypeofContact_imputed','ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		random_over_sampler = ImbalancedClass(dataframe=actual_df,
										 target=TestImbalancedClassdClass.target, 
										 type='OverSampling', 
										 method='Random',
										 sampling_strategy='auto',
										 random_state=42)


		expected_df = random_over_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")

		self.assertEqual(expected_df_minority_class_size, actual_df_majority_class_size)
		self.assertEqual(expected_df_majority_class_size, expected_df_minority_class_size)



	def test_CondensedNearestNeighbour(self):
		print("\nCondensedNearestNeighbour test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		condensed_nearest_neighbour_sampler = ImbalancedClass(dataframe=actual_df,
														 target=TestImbalancedClassdClass.target, 
														 type='UnderSampling', 
														 method='CondensedNearestNeighbour',
														 sampling_strategy='auto',
														 random_state=42)


		expected_df = condensed_nearest_neighbour_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")

		self.assertEqual(expected_df_minority_class_size, actual_df_minority_class_size)



	def test_TomekLinks(self):
		print("\nTomekLinks test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		tomeklinks_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='TomekLinks',
										sampling_strategy='auto')


		expected_df = tomeklinks_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")

		# self.assertEqual(expected_df_minority_class_size, actual_df_minority_class_size)

	
	def test_OneSidedSelection(self):
		print("\nOneSidedSelection test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		onesidedselection_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='OneSidedSelection',
										sampling_strategy='auto')


		expected_df = onesidedselection_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")

		# self.assertEqual(expected_df_minority_class_size, actual_df_minority_class_size)	


	def test_EditedNearestNeighbours(self):
		print("\nEditedNearestNeighbours test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		edited_nearest_neighbours_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='EditedNearestNeighbours',
										sampling_strategy='auto')


		expected_df = edited_nearest_neighbours_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")

		# self.assertEqual(expected_df_minority_class_size, actual_df_minority_class_size)	

	def test_RepeatedEditedNearestNeighbours(self):
		print("\nRepeatedEditedNearestNeighbours test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		repeated_edited_nearest_neighbours_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='RepeatedEditedNearestNeighbours',
										sampling_strategy='auto')


		expected_df = repeated_edited_nearest_neighbours_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")


	def test_AllKNN(self):
		print("\nAllKNN test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		allKNN_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='AllKNN',
										sampling_strategy='auto')


		expected_df = allKNN_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")




	def test_NeighbourhoodCleaningRule(self):
		print("\nNeighbourhoodCleaningRule test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		neighbourhood_cleaning_rule_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='NeighbourhoodCleaningRule',
										sampling_strategy='auto')


		expected_df = neighbourhood_cleaning_rule_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")

	def test_NearMiss(self):
		print("\nNearMiss test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		near_miss_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='NearMiss',
										sampling_strategy='auto')


		expected_df = near_miss_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")


	def test_InstanceHardnessThreshold(self):
		print("\nInstanceHardnessThreshold test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		InstanceHardnessThreshold_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='UnderSampling', 
										method='InstanceHardnessThreshold',										
										sampling_strategy='auto',
										random_state=42)


		expected_df = InstanceHardnessThreshold_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")



	def test_SMOTE(self):
		print("\nSMOTE test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		smote_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='OverSampling', 
										method='SMOTE',										
										sampling_strategy='auto',
										random_state=42,
										k_neighbors=5)


		expected_df = smote_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")



	def test_SMOTENC(self):
		print("\nSMOTENC test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'TypeofContact_imputed','ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		smotenc_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='OverSampling', 
										method='SMOTENC',
										categorical_features=[2],										
										sampling_strategy='auto',
										random_state=42,
										k_neighbors=5)


		expected_df = smotenc_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")



	def test_SMOTEN(self):
		print("\nSMOTEN test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Occupation','TypeofContact_imputed','ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		smoten_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='OverSampling', 
										method='SMOTEN',
										sampling_strategy='auto',
										random_state=42,
										k_neighbors=5)


		expected_df = smoten_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")
		# print("Expected dataframe: \n",expected_df.head(4))



	def test_ADASYN(self):
		print("\nADASYN test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		adasyn_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='OverSampling', 
										method='ADASYN',
										sampling_strategy='auto',
										random_state=42,
										n_neighbors=5)


		expected_df = adasyn_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")
		# print("Expected dataframe: \n",expected_df.head(4))


	def test_BorderlineSMOTE(self):
		print("\nBorderlineSMOTE test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		borderlinesmote_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='OverSampling', 
										method='BorderlineSMOTE',
										sampling_strategy='auto',
										random_state=42,
										k_neighbors=5,
										m_neighbors=10)


		expected_df = borderlinesmote_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")
		# print("Expected dataframe: \n",expected_df.head(4))



	def test_SVMSMOTE(self):
		print("\nSVMSMOTE test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		svmsmote_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='OverSampling', 
										method='SVMSMOTE',
										sampling_strategy='auto',
										random_state=42,
										k_neighbors=5,
										m_neighbors=10)


		expected_df = svmsmote_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")
		# print("Expected dataframe: \n",expected_df.head(4))



	def test_KMeansSMOTE(self):
		print("\nKMeansSMOTE test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		kmeans_smote_sampler = ImbalancedClass(dataframe=actual_df,
										target=TestImbalancedClassdClass.target, 
										type='OverSampling', 
										method='KMeansSMOTE',
										sampling_strategy='auto',
										random_state=42,
										k_neighbors=2)


		expected_df = kmeans_smote_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")
		# print("Expected dataframe: \n",expected_df.head(4))



	def test_SMOTEENN(self):
		print("\nSMOTEENN test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		smoteenn_sampler = ImbalancedClass(dataframe=actual_df,
									 target=TestImbalancedClassdClass.target, 
									 type='Combine', 
									 method='SMOTEENN',
									 sampling_strategy='auto',
									 random_state=42)


		expected_df = smoteenn_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")
		# print("Expected dataframe: \n",expected_df.head(4))


	def test_SMOTETomek(self):
		print("\nSMOTETomek test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		smotetomek_sampler = ImbalancedClass(dataframe=actual_df,
									 target=TestImbalancedClassdClass.target, 
									 type='Combine', 
									 method='SMOTETomek',
									 sampling_strategy='auto',
									 random_state=42)


		expected_df = smotetomek_sampler.get_resampled_dataframe()
		expected_df_size, _ = expected_df.shape

		expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")


# -------------------------

	def test_EasyEnsembleClassifier(self):
		print("\nEasyEnsembleClassifier test")
		actual_df = pd.read_csv(TestImbalancedClassdClass.filename)
		selected_features = ['Age_imputed','MonthlyIncome_clean_imputed', 'ProdTaken']
		actual_df = actual_df[selected_features]
		actual_df_majority_class_size , actual_df_minority_class_size = self._get_target_distribution(actual_df)

		print(f"Distribution of target in actual   df: {actual_df_majority_class_size , actual_df_minority_class_size }")

		# print("Actual dataframe: \n",actual_df.head(4))

		# easyensembleclassifier_sampler = ImbalancedClass(dataframe=actual_df,
		# 							 target=TestImbalancedClassdClass.target, 
		# 							 type='Ensemble', 
		# 							 method='EasyEnsembleClassifier',
		# 							 sampling_strategy='auto',
		# 							 random_state=42)

		
		cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

		model = EasyEnsembleClassifier(n_estimators=10, 
														   base_estimator=None, 
														   warm_start=False, 
														   sampling_strategy='auto', 
														   replacement=False, 
														   n_jobs=None, 
														   random_state=42,
														   verbose=0)


		X = actual_df[selected_features].drop('ProdTaken', axis=1)
		y = actual_df['ProdTaken']
		scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)

		print('ROC AUC scores', scores)


		print('Mean ROC AUC: %.3f' % np.mean(scores))

		# expected_df = easyensembleclassifier_sampler.get_resampled_dataframe()
		# expected_df_size, _ = expected_df.shape

		# expected_df_majority_class_size, expected_df_minority_class_size = self._get_target_distribution(expected_df)

		# print(f"Distribution of target in expected df: {expected_df_majority_class_size , expected_df_minority_class_size }")



unittest.main()


