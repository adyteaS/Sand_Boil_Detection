import numpy as np
import pandas as pd
import pickle as pk
import math
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef


def main():

	np.set_printoptions(threshold=np.inf) #to print infinite array
	# read the training data file
	train_df = pd.read_csv('features_collected_train_956posneg_1912.txt', header=None)

	# column 0 is class, class label "1" represents positive sample.
	train = train_df.as_matrix()

	y = train[:,0]
	X = train[:,1:]

	#################################  Perform Coarse Search for gamma ##############################################
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	#c_range = np.linspace(-5,15,num=11)
	#C_range = [math.pow(2,i) for i in c_range]
	g_range = np.linspace(-15,3,num=10)
	gamma_range = [math.pow(2,j) for j in g_range]
	print("searched gamma_range")
	print(gamma_range)
	param_grid = dict(gamma=gamma_range, C=[1.0])
	grid_coarse = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1) # to run in parallel using all available cores put n_jobs=-1
	grid_coarse.fit(X,y)
	print("The best parameters for coarse search for gamma are %s with a score of %0.4f"% (grid_coarse.best_params_, grid_coarse.best_score_))

	print()
	print()


	################################ Perform Fine Search for gamma ################################################
	#C_best_coarse = math.log2(grid_coarse.best_params_['C'])
	gamma_best_coarse = math.log2(grid_coarse.best_params_['gamma'])
	#c_range = np.linspace(C_best_coarse-2, C_best_coarse+2, num=17)
	#C_range = [math.pow(2,i) for i in c_range]
	g_range = np.linspace(gamma_best_coarse-2, gamma_best_coarse+2, num=17)
	gamma_range = [math.pow(2,j) for j in g_range]
	print("searched gamma_range")
	print(gamma_range)
	param_grid = dict(gamma=gamma_range, C=[1.0])
	grid_fine = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1)
	grid_fine.fit(X,y)
	print("The best parameters for fine search for gamma are %s with a score of %0.4f"% (grid_fine.best_params_, grid_fine.best_score_))
	
	print()
	print()
	
	#################################  Perform Coarse Search for C value ##############################################
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	c_range = np.linspace(-5,15,num=11)
	C_range = [math.pow(2,i) for i in c_range]
	#g_range = np.linspace(-15,3,num=10)
	#gamma_range = [math.pow(2,j) for j in g_range]
	print("searched C_range")
	print(C_range)
	param_grid = dict(gamma=[grid_fine.best_params_['gamma']], C=C_range)
	grid_coarse = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1) # to run in parallel using all available cores put n_jobs=-1
	grid_coarse.fit(X,y)
	print("The best parameters for coarse search for C param are %s with a score of %0.4f"% (grid_coarse.best_params_, grid_coarse.best_score_))

	print()
	print()

	################################ Perform Fine Search for C value ################################################
	C_best_coarse = math.log2(grid_coarse.best_params_['C'])
	#gamma_best_coarse = math.log2(grid_coarse.best_params_['gamma'])
	c_range = np.linspace(C_best_coarse-2, C_best_coarse+2, num=17)
	C_range = [math.pow(2,i) for i in c_range]
	#g_range = np.linspace(gamma_best_coarse-2, gamma_best_coarse+2, num=17)
	#gamma_range = [math.pow(2,j) for j in g_range]
	print("searched C_range")
	print(C_range)
	param_grid = dict(gamma=[grid_fine.best_params_['gamma']], C=C_range)
	grid_fine = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=-1)
	grid_fine.fit(X,y)
	print("The best parameters for fine search for C param are %s with a score of %0.4f"% (grid_fine.best_params_, grid_fine.best_score_))
	

	############################# Run 10-fold with best C and Gamma #################
	clf = SVC(C=grid_fine.best_params_['C'],kernel='rbf',gamma=grid_fine.best_params_['gamma'])
	#clf = SVC(C=608.8740428813932, kernel='rbf', gamma=0.00048828125, probability=True)
	predicted = cross_val_predict(clf, X, y, cv=10, n_jobs=-1)
	
	print("svc_predicted")

	clf = SVC(probability=True)
	prob_predict = cross_val_predict(clf, X, y, cv=10, n_jobs=-1, method='predict_proba')
	#predicted = cross_val_predict(clf, X, y, cv=10)

	confusion = confusion_matrix(y, predicted)
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
	MCC_cla = matthews_corrcoef(y, predicted)
	F1_cla = f1_score(y, predicted)
	PREC_cla = precision_score(y, predicted)
	REC_cla = recall_score(y, predicted)
	Accuracy_cla = accuracy_score(y, predicted)
	print('TP = ', TP, file=open("SVM_10fcv_full_956pos_956neg.txt", "w"))
	print('TN = ', TN, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('FP = ', FP, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('FN = ', FN, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('Recall/Sensitivity = %.5f' %REC_cla, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('Specificity = %.5f' %SPE_cla, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('Accuracy_Balanced = %.5f' %ACC_Bal, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('Overall_Accuracy = %.5f' %Accuracy_cla, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('FPR_bag = %.5f' %FPR, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('FNR_bag = %.5f' %FNR, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('Precision = %.5f' %PREC_cla, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('F1 = %.5f' % F1_cla, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	print('MCC = %.5f' % MCC_cla, file=open("SVM_10fcv_full_956pos_956neg.txt", "a"))
	
	pred = np.column_stack([prob_predict])
	print(np.column_stack([pred]), file=open("SVM_10fcv_full_956pos_956negprob.txt","w"))
	

if __name__ == '__main__':
    main()

