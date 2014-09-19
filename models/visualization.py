import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def make_roc_curve(test_outcomes, probs, **kwargs):
    """
    test_outcomes should be an array of true labels
    probs is an array of probabilities of belonging to class 1
    Plots FPR vs TPR
    Returns fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = roc_curve(test_outcomes, probs, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc, **kwargs)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specificity)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="best")
    return fpr, tpr, thresholds

def make_recall_specificity_curve(test_outcomes, probs, **kwargs):
    """
    test_outcomes is an array of true labels
    probs is an array of probabilities for each item belonging to class one
    Plots recall (x-axis) vs precision (y-axis)
    Returns precision, recall, thresholds
    """
    precision, recall, thresholds = precision_recall_curve(test_outcomes, probs)
    area = auc(recall, precision)
    plt.plot(recall, precision, label='Precision-Recall Curve (area = {:.2f}'.format(area), **kwargs)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall-Precision Curve')
    plt.legend(loc='best')
    return precision, recall, thresholds

def plot_feature_importance(fitted_rf_model, feature_cols):

    print("Feature ranking:")

    importances = fitted_rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in fitted_rf_model.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    num_features = len(indices)

    for rank, ind in zip(np.arange(num_features), indices):
        print '{}. {} - {:.3f}'.format(rank+1, feature_cols[ind], importances[ind])

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(num_features), np.array(feature_cols)[indices], rotation=270)
    plt.xlim([-1, num_features])
    plt.ylim([0, 0.5])
    plt.show()

