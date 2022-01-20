from tensorflow.keras import backend as K
import deb
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
from icecream import ic
import sklearn
import matplotlib.pyplot as plt
from icecream import ic
import pathlib
class Metrics():

	def __init__(self, paramsTrain):
		self.paramsTrain = paramsTrain

			
	def filterSamples(self, prediction, label, class_n):
		prediction=prediction[label<class_n] #logic
		label=label[label<class_n] #logic
		return prediction, label

	def plotROCCurve(self, y_test, y_pred, modelId, nameId, unknown_class_id = 39, pos_label=0):
		print("y_test.shape", y_test.shape)
		print("y_pred.shape", y_pred.shape)
		print("y_test.dtype", y_test.dtype)
		print("y_pred.dtype", y_pred.dtype)
		deb.prints(np.unique(y_test))   
		deb.prints(np.unique(y_pred))
		y_test = y_test.copy()
		y_test[y_test!=unknown_class_id] = 0
		y_test[y_test==unknown_class_id] = 1
		deb.prints(np.unique(y_test))   
		deb.prints(np.unique(y_pred))

		# =========================== Get metric value


		fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred, pos_label=pos_label)
#        roc_auc = metrics.auc(tpr, fpr)
		roc_auc = sklearn.metrics.auc(fpr, tpr)

		deb.prints(roc_auc)
		deb.prints(thresholds)
		deb.prints(tpr)
#        pdb.set_trace()

		optimal_idx = np.argmax(tpr - fpr)
		#optimal_idx = np.argmax(fpr - tpr)
		
		optimal_threshold = thresholds[optimal_idx]
		deb.prints(optimal_threshold)

		# =================== Find thresholds for specified TPR value
		tpr_threshold_values = [0.1, 0.3, 0.5, 0.7, 0.9]
		deb.prints(tpr_threshold_values)
		tpr_idxs = [np.where(tpr>tpr_threshold_values[0])[0][0],
			np.where(tpr>tpr_threshold_values[1])[0][0],
			np.where(tpr>tpr_threshold_values[2])[0][0],
			np.where(tpr>tpr_threshold_values[3])[0][0],
			np.where(tpr>tpr_threshold_values[4])[0][0]
		]
		deb.prints(tpr_idxs)
		
		thresholds_by_tpr = thresholds[tpr_idxs]
		deb.prints(thresholds_by_tpr)
#        pdb.set_trace()
		# ========================== Plot
		pathlib.Path("results/open_set/roc_curve/").mkdir(parents=True, exist_ok=True)

		np.savez("results/open_set/roc_curve/roc_curve_"+modelId+"_"+nameId+".npz", fpr=fpr, tpr=tpr)
		plt.figure()
		plt.plot([0, 1], [0, 1], 'k--')
#        plt.plot(tpr, fpr, label = 'AUC = %0.2f' % roc_auc)
		plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('AUC = %0.2f' % roc_auc)
		plt.savefig('roc_auc_'+modelId+"_"+nameId+'.png', dpi = 500)
#        plt.gca().set_aspect('equal', adjustable='box')
		#plt.show()
		return optimal_threshold, fpr, tpr, roc_auc

class MetricsTranslated(Metrics):
	def filterSamples(self, prediction, label, class_n):
		prediction=prediction[label!=255] #logic
		label=label[label!=255] #logic
		return prediction, label
