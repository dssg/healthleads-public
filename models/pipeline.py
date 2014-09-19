import numpy as np
import pandas as pd
import statsmodels.api as sm
from models.features import get_need_desk_disconnect_dict, get_need_composite_index
from models.features import get_desk_need_dict, get_sum_need_days_per_case
from models.diagnostics import get_diagnostics

def make_testing_training(data, percent_training, random_split=False, seed=None):
    """
    Returns a testing and training set
    """
    ## Making testing and training sets
    data['computed Case Date/Time Closed'] = pd.to_datetime(data['Case Date/Time Closed'])
    ordered_data = data.sort(columns=['computed Case Date/Time Closed'])
    np.random.seed(seed=seed) 
    nrows, ncols = ordered_data.shape

    if random_split:
        training_indices = np.random.choice(ordered_data.index, size=int(nrows*percent_training), replace=False)
        training = ordered_data.ix[training_indices]
        testing = ordered_data[~data['case_id'].isin(training['case_id'])]
    else: # split by date
        training_stop_index = int(percent_training * nrows)
        training = ordered_data[:training_stop_index]
        testing = ordered_data[training_stop_index:]

    return training, testing

def add_features_to_training_testing(training, testing, need, desk_names):
    """
    Returns training, testing with the following features:
    1. p_all_needs_disconnect
    2. sum_mean_days
    3. sum_median_days
    """
    filtered_need = need[need['Need Desk Location'].isin(desk_names)]
    need_types = filtered_need['Need Type: Category'].unique()
    indexed_need = need.set_index(keys=['Case ID'])
    indexed_need_by_need_desk = need.set_index(keys=['Need Type: Category', 'Need Desk Location'])
    indexed_need_training = filtered_need[filtered_need['Case ID'].isin(training['case_id'])].set_index('Need Desk Location')
    p_desk_need = get_need_desk_disconnect_dict(need_types, desk_names, indexed_need_training, indexed_need_by_need_desk)
    need_indexed_by_case = filtered_need.set_index('Case ID')
    
    # Creating p_all_needs_disconnect
    training['p_all_needs_disconnect'] = training.apply(get_need_composite_index, args=(p_desk_need, need_indexed_by_case), axis=1)
    testing['p_all_needs_disconnect'] = testing.apply(get_need_composite_index, args=(p_desk_need, need_indexed_by_case), axis=1)

    # Creating sum_median_days and sum_mean_days
    assert 'Days Until Need Closed' in need.columns, "You must create 'Days Until Need Closed' in need" 
    desk_need_dict = get_desk_need_dict(desk_names, need_types, indexed_need_training, indexed_need, 'Days Until Need Closed')
    training['sum_mean_days'], training['sum_median_days'] = zip(*training.apply(get_sum_need_days_per_case, args=(desk_need_dict, need_indexed_by_case), axis=1).values)
    testing['sum_mean_days'], testing['sum_median_days'] = zip(*testing.apply(get_sum_need_days_per_case, args=(desk_need_dict, need_indexed_by_case), axis=1).values)

    return training, testing

def run_random_forest(rf_clf, training, testing, feature_cols, outcome_col):
    """
    Returns fitted_rf_model, diagnostics, predicted_rf_probs
    """
    X_train, X_test = training[feature_cols].values, testing[feature_cols].values
    Y_train, Y_test = training[outcome_col].values, testing[outcome_col].values
    fitted_rf_model = rf_clf.fit(X_train, Y_train)
    rf_diagnostics = get_diagnostics(testing[outcome_col], testing[feature_cols], fitted_rf_model, 'rf')
    predicted_rf_probs = [p[1] for p in fitted_rf_model.predict_proba(X_test)]

    return fitted_rf_model, rf_diagnostics, predicted_rf_probs

def run_logistic_regression(training, testing, feature_cols, outcome_col):
    """
    Returns fitted_logit_model, logit_diagnostics, predicted_logit_probs
    """
    if 'intercept' not in training.columns:
        training['intercept'] = 1
    if 'intercept' not in testing.columns:
        testing['intercept'] = 1
    intercept_feature_cols = feature_cols + ['intercept']
    logit = sm.Logit(training[outcome_col], training[intercept_feature_cols])
    fitted_logit_model = logit.fit()
    logit_diagnostics = get_diagnostics(testing[outcome_col], testing[intercept_feature_cols], fitted_logit_model, model_type = 'logit')
    predicted_logit_probs = fitted_logit_model.predict(testing[intercept_feature_cols])

    return fitted_logit_model, logit_diagnostics, predicted_logit_probs
