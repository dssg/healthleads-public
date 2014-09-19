import pandas as pd

def get_diagnostics(true_outcomes, features, fitted_model, model_type,  threshold=0.5):
    """
    Returns a dictionary of key-value pairs of true pos, true neg, false pos, false neg
    and each correspoding percentage
    """
    diagnostics = {}
                    
    total_data_points = len(true_outcomes)

    if model_type == 'logit':
    	predicted_probs = fitted_model.predict(features)
    	predicted_outcomes = pd.Series(predicted_probs, index=true_outcomes.index).apply(lambda val: 1 if val >= threshold else 0)
    elif model_type in ['rf','SVM','NB']:
    	predicted_outcomes = pd.Series(fitted_model.predict(features), index = true_outcomes.index)
    else:
    	raise ValueError("Model type must be specified.")

    comparison = pd.DataFrame({'true_outcome': true_outcomes, 
                               'predicted_outcomes': predicted_outcomes})
    
    get_true_positives = lambda row: row['predicted_outcomes'] == 1 and row['true_outcome'] == 1
    get_true_negatives = lambda row: row['predicted_outcomes'] == 0 and row['true_outcome'] == 0
    get_false_positives = lambda row: row['predicted_outcomes'] == 1 and row['true_outcome'] == 0
    get_false_negatives = lambda row: row['predicted_outcomes'] == 0 and row['true_outcome'] == 1
    
    true_positive = comparison.apply(get_true_positives, axis=1)
    true_negative = comparison.apply(get_true_negatives, axis=1)
    false_positive = comparison.apply(get_false_positives, axis=1)
    false_negative = comparison.apply(get_false_negatives, axis=1)
    
    diagnostics['sensitivity/recall/tpr'] = float(sum(true_positive)) / (sum(true_positive) + sum(false_negative)) 
    diagnostics['accuracy'] = float(sum(true_positive) + sum(true_negative))/total_data_points
    diagnostics['true negative rate'] = float(sum(true_negative))/(sum(false_positive) + sum(true_negative))
    diagnostics['false positive rate'] = float(sum(false_positive))/(sum(false_positive) + sum(true_negative))
    diagnostics['false negative rate'] = float(sum(false_negative))/(sum(true_positive) + sum(false_negative))
    diagnostics['precision'] = float(sum(true_positive))/(sum(true_positive) + sum(false_positive))
    diagnostics['f1'] = float(2 * sum(true_positive))/(2*sum(true_positive) + sum(false_positive) + sum(false_negative)) 

    return diagnostics
