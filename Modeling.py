
# coding: utf-8

# ## Importing and Setup

# In[1]:

import time
from datetime import datetime, timedelta
import statsmodels.api as sm
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from prepare_data.load import cleaned_data, need, activity, client, case, advocate_demographics, advocate
from models.pipeline import run_random_forest, run_logistic_regression
from models.pipeline import make_testing_training, add_features_to_training_testing
from models.features import get_contacts_within_time, get_time_difference
from models.features import get_need_desk_success_dict, get_need_composite_index
from models.features import get_desk_need_dict,get_sum_need_days_per_case,get_time_delta,get_time_difference
from models.features import get_contacts_within_date, get_outcomes_from_cases
from models.diagnostics import get_diagnostics
from models.visualization import  make_recall_specificity_curve, make_roc_curve, plot_feature_importance


# In[2]:

# A little bit of cleanup...
filtered_activity = activity[activity['Case ID'].isin(cleaned_data['case_id'])]
filtered_need = need[need['Case ID'].isin(cleaned_data['case_id'])]


# In[ ]:

window_days = 30
data = cleaned_data
data = data[data['Days Until Case Closed'] > window_days]


# ## Part A: Random Forest
# Using a Random Forest to model whether a patient will disconnect from Health Leads or not.

# ### A1 - Features
# Feature selection has already been performed via inspecting previous RF models. Below are the features that were the most significant.

# In[18]:

outcome_col = 'binary_outcome'
feature_cols = [
                'num_contacts_within30',
                'num_attempts_within30',
                'num_contacts_within7',
                'num_attempts_within7',
                'num_contacts_within15',
                'num_attempts_within15',
                'p_all_needs_disconnect',
                'sum_median_days',
                ]


# In[14]:

rf_clf = RandomForestClassifier(n_estimators=500, max_depth=None)


# ### A2: Manually Separating Test and Training Sets

# We separate the testing and training data by a particular date, using
# 80% of the data to predict the future 20% of cases.

# In[6]:

desk_names = cleaned_data['Case Desk Location'].unique()


# In[7]:

# Creating a column in the need dataframe that is necessary to add the relevant features to the data
need_date_opened_format = '%m/%d/%Y %H:%M'
need_date_closed_format = '%m/%d/%Y %H:%M'
need['Days Until Need Closed'] = need.apply(get_time_difference, args = ('Need Date Opened', 'Need Date Closed', need_date_opened_format, need_date_closed_format), axis = 1)


# In[50]:

training, testing = make_testing_training(data, percent_training=0.80, random_split=False)


# In[51]:

training, testing = add_features_to_training_testing(training, testing, need, desk_names)


# In[4]:

training.binary_outcome.value_counts(normalize=True)
# 0    0.591141
# 1    0.408859


# In[5]:

testing.binary_outcome.value_counts(normalize=True)
# 0    0.631175
# 1    0.368825


# ### A3 -- Running the Random Forest

# In[52]:

fitted_rf_model, rf_diagnostics, rf_predicted_probs = run_random_forest(rf_clf, training, testing, feature_cols, outcome_col)


# ### A4 -- Diagnostics
# We examine the ROC and Precision-Recall Curves. Optimally, we'd like to find a point on the Precision-Recall curve that has high precision (>75%) for decent recall.

# In[23]:

fpr, tpr, thresholds = make_roc_curve(testing[outcome_col], rf_predicted_probs, linewidth=2)


# In[24]:

precision, recall, thresholds = make_recall_specificity_curve(testing[outcome_col], rf_predicted_probs, linewidth=2)


# ### A5 -- Diagnosing Feature Importance

# In[25]:

plot_feature_importance(fitted_rf_model, feature_cols)


# It makes sense that the needs compexity index is one of the most significant features.
# 
# However, it's not clear why the median number of days for all needs is significant.

# ## Part B: Using Logistic Regression

# ### B1 -- Running the LR

# In[53]:


fitted_logit_model, logit_diagnostics, predicted_logit_probs = run_logistic_regression(training, testing, feature_cols, outcome_col)


# ### B2 -- Diagnosing the LR

# In[27]:

logit_diagnostics


# In[28]:

fitted_logit_model.summary()


# In[29]:

fpr, tpr, thresholds = make_roc_curve(testing[outcome_col], predicted_logit_probs)


# In[30]:

precision, recall, thresholds = make_recall_specificity_curve(testing[outcome_col], predicted_logit_probs)


# ## Part C: Feature Importance Over Time

# Looking at the importance of a client picking up the phone vs. not picking up over time

# In[32]:

window_range = range(1, 51)
params = {num: 
           {'ratio': 0, 
            'num_contacts_within{}'.format(num): 0, 
            'num_attempts_within{}'.format(num): 0,}
           for num in window_range}
data = cleaned_data[cleaned_data['Days Until Case Closed'] > window_range[-1]]


# In[33]:

training, testing = make_testing_training(data, 0.8)


# In[34]:

desk_names = data['Case Desk Location'].unique()


# In[35]:

training, testing = add_features_to_training_testing(training, testing, need, desk_names)


# In[ ]:

for num in window_range:
    feature_cols = [
     'num_contacts_within{}'.format(num),
     'num_attempts_within{}'.format(num),
     'Client Age',
     'p_all_needs_disconnect',
     'sum_median_days',]
    fitted_logit_model, logit_diagnostics, pred_logit_probs = run_logistic_regression(training, testing, feature_cols, outcome_col)
    num_contacts_within_param = fitted_logit_model.params['num_contacts_within{}'.format(num)]
    num_attempts_within_param = fitted_logit_model.params['num_attempts_within{}'.format(num)]
    ratio = float(num_contacts_within_param)/num_attempts_within_param
    params[num]['ratio'] = ratio
    params[num]['num_contacts_within{}'.format(num)] = num_contacts_within_param
    params[num]['num_attempts_within{}'.format(num)] = num_attempts_within_param


# In[37]:

contacts = [params[num]['num_contacts_within{}'.format(num)] for num in window_range]


# In[38]:

attempts = [params[num]['num_attempts_within{}'.format(num)] for num in window_range]


# In[42]:

plt.title('Impact on Disconnection')
plt.plot(window_range, [1 - exp(c) for c in contacts], 'g', linewidth = 2, label = 'Successful Phone Calls')
plt.plot(window_range, [exp(a) - 1 for a in attempts], 'b', linewidth = 2, label = 'Unsuccessful Phone Calls')
plt.xlabel('Days Since Case Opened')
plt.ylabel('% Impact')
plt.legend(loc='best')
fig = plt.gcf()


# As we brought up in our final presentation to Health Leads, this graph shows that the importance of successful phone calls trumps those of unsuccessful phone calls throughout the entire client relationship.
# 
# Given that, we suggested Health Leads to make more frequent calls in the beginning stages of a relationship.

# ## Part D: Ensemble Logitistic Regression & Random Forest

# In[43]:

# Let's try a combination of the RF & Logistic Regression
logit_weights = arange(0, 1.1, 0.1)


# In[44]:

ensemble_diagnostics = {weight: {
                                 'roc': 0, 
                                 'precision-recall': 0} 
                        for weight in logit_weights}


# In[54]:

for weight in logit_weights:
    ensemble_probs = weight*predicted_logit_probs + (1 - weight)*np.array(rf_predicted_probs)
    fpr, tpr, thresholds = roc_curve(testing[outcome_col], ensemble_probs, pos_label = 1)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(testing[outcome_col], ensemble_probs)
    precision_recall_auc = auc(recall, precision)
    ensemble_diagnostics[weight]['roc'] = roc_auc
    ensemble_diagnostics[weight]['precision-recall'] = precision_recall_auc


# In[55]:

plt.plot(logit_weights, [ensemble_diagnostics[weight]['roc'] for weight in logit_weights])


# In[59]:

plt.plot(logit_weights, [ensemble_diagnostics[weight]['precision-recall'] for weight in logit_weights])


# It seems like the most predictive model would be a combination of a Random Forest and Logistic Regression, with the most weight placed on the Random Forest.
