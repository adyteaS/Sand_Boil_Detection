# Code for stacking of various methods using 3-fold CV

import numpy as np
import pandas as pd
import pickle as pk
import math
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef


def main():
	# read the training data file for window size 11
	train_df_1 = pd.read_csv('features_selected.csv', header=None)
	train_1 = train_df_1.as_matrix()
	y_1 = train_1[:,0]
	X_1 = train_1[:,1:]
	scaler = StandardScaler()
	X_scale_1 = scaler.fit_transform(X_1)


	########################## First base layer ML method is Random Forest ########################
	clf = RandomForestClassifier(n_estimators=1000)
	pred_rand = cross_val_predict(clf, X_scale_1, y_1, cv=10, n_jobs=-1, method='predict_proba') # cross val predict inherently does stratified k-fold
	y_pred_rand = np.column_stack([pred_rand])
	print("rand_predicted")

	################################ First base layer-> LogisticRegression
	clf = LogisticRegression()
	pred_logreg = cross_val_predict(clf, X_scale_1, y_1, cv=10, n_jobs=-1, method='predict_proba') # cross val predict inherently does stratified k-fold
	y_pred_logreg = np.column_stack([pred_logreg])
	print("logreg_predicted")

	########################### Second base layer ML method is GBC ###############################
	# clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=3)
	# pred_gbc = cross_val_predict(clf, X_scale, y, cv=10, n_jobs=-1, method='predict_proba') # cross val predict inherently does stratified k-fold
	# y_pred_gbc = np.column_stack([pred_gbc])
	# print("gbc_predicted")

	# ######################### Second Base layer is ExtraTree##########################################
	# clf = ExtraTreesClassifier(n_estimators=1000)
	# pred_extratree = cross_val_predict(clf, X_scale_1, y_1, cv=3, n_jobs=-1, method = 'predict_proba')
	# y_pred_extratree = np.column_stack([pred_extratree])
	# print("extratree_predicted")


	########################## Third base layer ML method is knn ###############################
	# clf = KNeighborsClassifier(n_neighbors=7)
	# pred_knn = cross_val_predict(clf, X_scale_1, y_1, cv=3, n_jobs=-1, method='predict_proba') # cross val predict inherently does stratified k-fold
	# y_pred_knn = np.column_stack([pred_knn])
	# print("knn_predicted")

	# #################################### Fourth base layer is BaggingClassifier ###################
	# clf = BaggingClassifier(n_estimators=1000)
	# pred_bag = cross_val_predict(clf, X_scale_3, y_3, cv=10, n_jobs=-1)
	# y_pred_bag = np.column_stack([pred_bag])
	# print("bag_predicted")
	
	############################# Run 10-fold with best C and Gamma #################
	#clf = SVC(C=grid_fine.best_params_['C'],kernel='rbf',gamma=grid_fine.best_params_['gamma'])
	clf = SVC(C=1.4142135623730951, kernel='rbf', gamma=0.001953125, probability=True)
	pred_SVM = cross_val_predict(clf, X_scale_1, y_1, cv=10, n_jobs=-1, method='predict_proba')
	y_pred_SVM = np.column_stack([pred_SVM])
	print("svc_predicted")

	############################### Combine probabilities of base layer to the original features and run SVM ##############################
	X = np.column_stack([X_1, y_pred_rand, y_pred_logreg, y_pred_SVM])
	#X = np.column_stack([X_1, y_pred_rand[:,1], y_pred_logreg[:,1], y_pred_knn[:,1]])
	scaler = StandardScaler()
	X_scale_etc = scaler.fit_transform(X)

	print("output of base layer has been addded to the original features and is ready to be used in meta layer")


	# ########################### Meta layer ML method is Gradient Boosting Classifier ###############################
	# clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=3)
	# predicted = cross_val_predict(clf, X_scale_gbc, y_1, cv=3, n_jobs=-1) # cross val predict inherently does stratified k-fold
	# #y_pred_gbc = np.column_stack([predicted])
	# #np.savetxt(fname=out_file_logreg, X=y_pred, fmt='%0.6f %0.6f', header='probN, probB', comments='')
	
	######################### METAlayer is ExtraTree##########################################
	# clf = ExtraTreesClassifier(n_estimators=1000)
	# predicted = cross_val_predict(clf, X_scale_etc, y_1, cv=10, n_jobs=-1, method = 'predict_proba')
	#y_pred_extratree = np.column_stack([pred_extratree])
	#print("extratree_predicted")

	###########MEta layer is KNN###############################
	clf = KNeighborsClassifier(n_neighbors=7)
	predicted = cross_val_predict(clf, X_scale_etc, y_1, cv=10, n_jobs=-1) # cross val predict inherently does stratified k-fold
	#y_pred_knn = np.column_stack([pred_knn])
	#print("knn_predicted")

	# #################################  Perform Coarse Search ##############################################
	# c_range = np.linspace(-5,15,num=11)
	# C_range = [math.pow(2,i) for i in c_range]
	# g_range = np.linspace(-15,3,num=10)
	# gamma_range = [math.pow(2,j) for j in g_range]
	# print("searched C_range")
	# print(C_range)
	# print("searched gamma_range")
	# print(gamma_range)
	# param_grid = dict(gamma=gamma_range, C=C_range)
	# grid_coarse = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1) # to run in parallel using all available cores put n_jobs=-1
	# grid_coarse.fit(X_scale_SVM,y)
	# print("The best parameters for coarse search are %s with a score of %0.4f"% (grid_coarse.best_params_, grid_coarse.best_score_))

	# print()
	# print()


	# ################################ Perform Fine Search ################################################
	# C_best_coarse = math.log2(grid_coarse.best_params_['C'])
	# gamma_best_coarse = math.log2(grid_coarse.best_params_['gamma'])
	# c_range = np.linspace(C_best_coarse-2, C_best_coarse+2, num=17)
	# C_range = [math.pow(2,i) for i in c_range]
	# g_range = np.linspace(gamma_best_coarse-2, gamma_best_coarse+2, num=17)
	# gamma_range = [math.pow(2,j) for j in g_range]
	# print("searched C_range")
	# print(C_range)
	# print("searched gamma_range")
	# print(gamma_range)
	# param_grid = dict(gamma=gamma_range, C=C_range)
	# grid_fine = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1)
	# grid_fine.fit(X_scale_SVM,y)
	# print("The best parameters for fine search are %s with a score of %0.4f"% (grid_fine.best_params_, grid_fine.best_score_))


	# ############################# Run 10-fold with best C and Gamma #################
	# clf = SVC(C=grid_fine.best_params_['C'],kernel='rbf',gamma=grid_fine.best_params_['gamma'])
	# #clf = SVC(C=608.8740428813932, kernel='rbf', gamma=0.00048828125, probability=True)
	# predicted = cross_val_predict(clf, X_scale_SVM, y, cv=10, n_jobs=-1)
	# print("svc_predicted")

	#clf = svm.SVC()
	#predicted = cross_val_predict(clf, X, y, cv=10)
	
	
	
	confusion = confusion_matrix(y_1, predicted)
	print(confusion)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	# Specificity
	SPE_cla = (TN/float(TN+FP))

	# False Positive Rate
	FPR = (FP/float(TN+FP))

	#False Negative Rate (Miss Rate)
	FNR = (FN/float(FN+TP))

	#Balanced Accuracy
	ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))

	# compute MCC
	MCC_cla = matthews_corrcoef(y_1, predicted)
	F1_cla = f1_score(y_1, predicted)
	PREC_cla = precision_score(y_1, predicted)
	REC_cla = recall_score(y_1, predicted)
	Accuracy_cla = accuracy_score(y_1, predicted)
	print('TP = ', TP, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "w"))
	print('TN = ', TN, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('FP = ', FP, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('FN = ', FN, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('Recall/Sensitivity = %.5f' %REC_cla, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('Specificity = %.5f' %SPE_cla, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('Accuracy_Balanced = %.5f' %ACC_Bal, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('Overall_Accuracy = %.5f' %Accuracy_cla, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('FPR_bag = %.5f' %FPR, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('FNR_bag = %.5f' %FNR, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('Precision = %.5f' %PREC_cla, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('F1 = %.5f' % F1_cla, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))
	print('MCC = %.5f' % MCC_cla, file=open("stacking_RDFlogregSVM_metaKNN_selected.txt", "a"))

if __name__ == '__main__':
    main()







