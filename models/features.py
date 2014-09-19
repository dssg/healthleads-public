import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_outcomes_from_cases(row, reindexed_need):
    """
    Returns case-level  outcome
    """
    case_id = row['ID (Case Sensitive)']
    results = reindexed_need.ix[[case_id]]['Need Resolution Category']

    if any(results.isin(['Success'])):
        outcome = 4
    elif any(results.isin(['Equipped'])):
        outcome = 3
    elif any(results.isin(['Failure'])):
        outcome = 2
    elif all(results.isin(['Disconnection'])):
        outcome = 1
    else:
        outcome = 0
    return outcome

def get_num_contacts(case_id, outcome, reindexed_activity):
    """
    Given a case id and activity dataframe, returns a responsiveness score.
    outcome is binarilized already
    """
    results = reindexed_activity.ix[[case_id]] #passing a list ensures a dataframe is returned, even if there is only one row
    filtered_results = results[results['Type'] == 'Progress Note']
    value_counts = filtered_results['Sub Type'].value_counts()
    successful_contacts = value_counts.get('Contacted Client', 0)
    unsuccessful_contacts = value_counts.get('Attempted Contact', 0)
    if outcome == 0: # only adjusting if disconnected
        unsuccessful_contacts = unsuccessful_contacts - 3 if unsuccessful_contacts >= 3 else 0
    return successful_contacts, unsuccessful_contacts

def get_responsiveness_scores(row, case_id_col, adj_outcome_col, reindexed_activity):
    """
    Applied to a data frame, returns a Series of responsiveness scores
    """
    case_id = row[case_id_col]
    outcome = row[adj_outcome_col]
    successful_contacts, unsuccessful_contacts = get_num_contacts(case_id, outcome, reindexed_activity)
    if successful_contacts == 0 and unsuccessful_contacts == 0:
        resp_score = np.nan
    else:
        resp_score = float(successful_contacts)/(successful_contacts + unsuccessful_contacts)
    return resp_score

def get_adjusted_unsuccessful(row, outcome_col, num_unsuccess_col):
    """
    Returns adjusted number of unsuccesful contacts given column of outcome and num_unsuccess
    """
    outcome = row[outcome_col]
    num_unsuccess = row[num_unsuccess_col]
    if outcome == 0:
        adj_num_unsuccess = num_unsuccess - 3 if num_unsuccess >= 3 else 0
    else:
        adj_num_unsuccess = num_unsuccess
    return adj_num_unsuccess


def get_num_response(case_ids, reindexed_activity, n=10):
    """
    Return a DataFrame contain each case's number of successful contacts and number of unsuccessful contacts
    """
    assert n <= len(case_ids), 'n must be <= length of cases'
    num_success = pd.Series()
    num_unsuccess = pd.Series()
    for case_id in case_ids[:n]:
    	results = reindexed_activity[reindexed_activity.index == case_id].apply(lambda row: row['Type'] == 'Progress Note', axis = 1)
        successful_contacts = len(results[results['Sub Type'] == 'Contacted Client'])
        unsuccessful_contacts = len(results[results['Sub Type'] == 'Attempted Contact'])
        num_success.set_value(case_id, successful_contacts)
        num_unsuccess.set_value(case_id,unsuccessful_contacts)
    df = pd.DataFrame(data={'case_id':num_success.keys(),'num_success':num_success, 'num_unsuccess':num_unsuccess})
    return df

def calculate_number_of_needs(row, case_id_col, reindexed_need):
    """
    Given a case id, returns the number of needs associated
    """
    case_id = row[case_id_col]
    rows = reindexed_need.ix[[case_id]]
    return len(rows)

def calculate_number_of_dependents(row, reindexed_client):
    """
    Applied to a case dataframe
    Given a client id, returns the number of dependents
    """
    client_id = row['HL Client ID']
    if client_id in reindexed_client.index:
        results = reindexed_client.ix[[client_id]]['Is Main Client?']
        return results.value_counts().get('Dependent', 0)
    else: # client has no dependents
        return 0

def get_time_delta(start_date, end_date, start_format, end_format):
    """
    Given strings representing time, returns a timedelta object
    representing the time difference between two dates
    """
    time_delta = pd.to_datetime(end_date, end_format) - pd.to_datetime(start_date, start_format)
    return time_delta

def get_time_difference(row, start_col, end_col, start_format, end_format, unit='days'):
   """
   Returns a Series object of days
   Unit can be D for Days, or Y for Years
   """
   start_date = row[start_col]
   end_date = row[end_col]
   if pd.isnull(start_date) or pd.isnull(end_date):
       return np.nan
   else:
       time_delta = get_time_delta(start_date, end_date, start_format, end_format)
       if unit == 'days':
           return time_delta.days
       elif unit == 'years':
           return float(time_delta.days)/365

def get_adjusted_case_outcome(row, case_id_col, filtered_need):
    """
    Returns adjusted case outcome
    """
    case_id = row[case_id_col]
    results = filtered_need.ix[[case_id]]['Need Resolution Category']
    val_counts = results.value_counts()
    num_success_need = val_counts.get('Success', 0)
    num_disconnect_need = val_counts.get('Disconnection', 0)
    if num_success_need >= num_disconnect_need:
        adj_outcome = 1
    else:
        adj_outcome = 0
    return adj_outcome

def redefine_successful_contacts(row):
    """
    Returns whether a contact is successful or not for activity type: Progress Note, Administrative Message, Client Survey
    TODO: Refactor out hard coded names
    """
    activity_type = row['Type']
    sub_type = row['Sub Type']
    method = row['Contact Method']
    if activity_type == 'Progress Note':
        if method == 'Paper':
            return 0
        elif method == 'Text Message':
            if sub_type == 'Incoming Message':
                return 1
            elif sub_type == 'Outgoing Message':
                return np.nan
            else: #attempted contact, contacted client
                return 0
        else:
            if sub_type == 'Contacted Client':
                return 1
            elif sub_type == 'Attempted Contact':
                return 0
            else:
                return np.nan
    elif activity_type == 'Administrative Message':
        if sub_type =='Incoming Message':
            response = row['Additional Details'].lower()
            if response in ['start', 'yes']:
                return 1
            else:
                return 0
        else:
            return np.nan
    elif activity_type == 'Client Survey':
        if sub_type == 'Survey Response':
            return 1
        else:
            return np.nan
    else:
        return np.nan


def get_contacts_within_time(row, case_id_col, window_days, indexed_activity, contact_result_col):
    """
    Returns the num of successful contacts, num of attempted contacts, and whether the case is closed or not at a given time period, unit: days
    TODO: Refactor out hard coded column names
    To set these values as columns in a df, df['A'], df['B'] = zip(*df.apply(get_contacts_within_time...))
    """

    case_id = row[case_id_col]
    days_until_case_closed = row['Days Until Case Closed']
    case_open_date = datetime.strptime(row['Case Date Enrolled'], '%m/%d/%Y %I:%M %p')
    time_delta = timedelta(days=window_days)
    cutoff_date = case_open_date + time_delta

    result = indexed_activity.ix[[case_id]]
    result['c_date'] = result.apply(lambda row: datetime.strptime(row['Completed Date'], '%m/%d/%Y %H:%M') if pd.notnull(row['Completed Date']) else np.nan, axis=1)

    filtered_result = result[result['c_date'] < cutoff_date]

    val_counts = filtered_result[contact_result_col].value_counts()
    if len(val_counts) == 0:
        num_contacts = 0
        num_attempts = 0
    else:
        num_contacts = val_counts.get(1, 0)
        num_attempts = val_counts.get(0, 0)

    close_flag = days_until_case_closed <= window_days

    return [num_contacts, num_attempts, close_flag]

def get_contacts_within_date(row, case_id_col, cutoff_date, indexed_activity, contact_result_col):
    """
    Returns the num of successful contacts, num of attempted contacts, and whether the case is closed or not, how many days it has been open at a given date
    TODO: Refactor out hard coded column names
    """
    case_id = row[case_id_col]
    case_open_date = datetime.strptime(row['Case Date Enrolled'], '%m/%d/%Y %I:%M %p')
    case_close_date = datetime.strptime(row['Case Date/Time Closed'], '%m/%d/%Y %I:%M %p')

    result = indexed_activity.ix[[case_id]]
    result['c_date'] = result.apply(lambda row: datetime.strptime(row['Completed Date'], '%m/%d/%Y %H:%M') if pd.notnull(row['Completed Date']) else np.nan, axis=1)

    filtered_result = result[result['c_date'] < cutoff_date]

    val_counts = filtered_result[contact_result_col].value_counts()
    if len(val_counts) == 0:
        num_contacts = 0
        num_attempts = 0
        responsive = 0
    else:
        num_contacts = val_counts.get(1, 0)
        num_attempts = val_counts.get(0, 0)
        responsive = float(num_contacts)/(num_contacts+num_attempts)

    close_flag = case_close_date <= cutoff_date
    if close_flag:
        days_opened = row['Days Until Case Closed']
    else:
        days_opened = get_time_difference(row, 'Case Date Enrolled','Case Date/Time Closed','%m/%d/%Y %I:%M %p','%m/%d/%Y %I:%M %p')

    return [num_contacts, num_attempts, responsive, close_flag, days_opened]

def get_need_desk_success_dict(need_types, desk_names, indexed_need_training, indexed_need):
    """
    Creates a dict of {desk1: {need1: % need1 successful, need2:...}, desk2:...}
    """
    p_desk_need = {desk:
                   {need: 0 for need in need_types}
                   for desk in desk_names}
    for desk in desk_names:
        desk_results = indexed_need_training.ix[[desk]]
        for need in need_types:
            desk_needs = desk_results[desk_results['Need Type: Category'] == need]
            val_counts = desk_needs['Need Resolution Category'].value_counts()
            num_success_need =  val_counts.get('Success', 0)
            num_disconnect_need = val_counts.get('Disconnection', 0)
            if num_success_need == 0 and num_disconnect_need == 0: # need type doesn't exist, use the overall per need type as estimator
                all_needs = indexed_need[indexed_need['Need Type: Category'] == need]
                val_counts = all_needs['Need Resolution Category'].value_counts()
                num_success_need =  val_counts.get('Success', 0)
                num_disconnect_need = val_counts.get('Disconnection', 0)
            p_success = float(num_success_need)/(num_success_need + num_disconnect_need)
            p_desk_need[desk][need] = p_success

    return p_desk_need

def get_need_desk_disconnect_dict(need_types, desk_names, indexed_need_training, indexed_need_by_need_desk):
    p_desk_need = {desk:
                   {need: 0 for need in need_types}
                   for desk in desk_names}
    for desk in desk_names:
	desk_results = indexed_need_training.ix[[desk]]
	for need in need_types:
	    desk_needs = desk_results[desk_results['Need Type: Category'] == need]
	    num_total_need = len(desk_needs)
	    val_counts = desk_needs['Need Resolution Category'].value_counts()
	    num_disconnect_need = val_counts.get('Disconnection', 0)

	    if num_total_need == 0: # if per desk, this need does not exist, use the overall need disconnection rate
	        global_needs = indexed_need_by_need_desk.ix[[need]]
	        num_total_need = len(global_needs)
		val_counts = global_needs['Need Resolution Category'].value_counts()
		num_disconnect_need = val_counts.get('Disconnection', 0)

	    p_disconnect = float(num_disconnect_need)/(num_total_need)
	    p_desk_need[desk][need] = p_disconnect

    return p_desk_need

def get_need_composite_index(row, p_desk_need, need_indexed_by_case):
    case_id = row['case_id']
    need_type = need_indexed_by_case.ix[[case_id]]['Need Type: Category']
    desk = row['Case Desk Location']
    p_all_need_disconnect = 1

    for need in need_type:
        p_all_need_disconnect *=  p_desk_need[desk][need]

    return p_all_need_disconnect

def get_desk_need_dict(desk_names, need_types, indexed_need_training, indexed_need, days_col):
    """
    Returns a dictionary of median/mean number of days to success for each need at each desk
    indexed_need should be indexed on desk names
    """
    desk_need_dict = {desk_name:
                      {need_type:
                       {'mean': 0, 'median': 0, 'total_needs': 0}
                       for need_type in need_types}
                      for desk_name in desk_names
                      }
    for desk_name in desk_need_dict:
        desk_results = indexed_need_training.ix[[desk_name]]
        for need_type in need_types:
	    desk_need = desk_results[desk_results['Need Type: Category'] == need_type]
	    desk_need_success_days = desk_need[desk_need['Need Resolution Category'] == 'Success'][days_col].dropna()
	    if len(desk_need_success_days) == 0: # if there are no successful needs at this desk
	        overall_need = indexed_need[indexed_need['Need Type: Category'] == need_type]
		desk_need_success_days = overall_need[overall_need['Need Resolution Category'] == 'Success'][days_col].dropna()
            desk_need_dict[desk_name][need_type]['mean'] = np.average(desk_need_success_days)
            desk_need_dict[desk_name][need_type]['median'] = np.median(desk_need_success_days)
            desk_need_dict[desk_name][need_type]['total_needs'] = len(desk_need_success_days)
    return desk_need_dict

def get_sum_need_days_per_case(row, desk_need_dict, need_indexed_by_case):
    """
    To set these values as columns in a df, df['A'], df['B'] = zip(*df.apply(get_sum_need_days_per_case...))
    """
    sum_mean_days = 0
    sum_median_days = 0
    case_id = row['case_id']
    desk_name = row['Case Desk Location']
    val_counts = need_indexed_by_case.ix[[case_id]]['Need Type: Category'].value_counts()
    for need_type in val_counts.keys():
        sum_mean_days += desk_need_dict[desk_name][need_type]['mean']
        sum_median_days += desk_need_dict[desk_name][need_type]['median']

    return pd.Series({'mean': sum_mean_days, 'median': sum_median_days})

def get_responsiveness_within(row, window_days):
    successful_contacts = row['num_contacts_within{}'.format(window_days)]
    attempted_contacts = row['num_attempts_within{}'.format(window_days)]
    resp_score = float(successful_contacts)/(successful_contacts + attempted_contacts)
    return resp_score

def get_needs_per_case(row, indexed_need):
    case_id = row['case_id']
    case_needs = indexed_need.ix[[case_id]]
    need_categories = case_needs['Need Type: Category']
    val_count = need_categories.value_counts()
    adult_education = val_count.get('Adult Education', 0)
    child_related = val_count.get('Child Related', 0)
    commodities = val_count.get('Commodities', 0)
    employment = val_count.get('Employment', 0)
    financial = val_count.get('Financial', 0)
    food = val_count.get('Food', 0)
    health = val_count.get('Health', 0)
    housing = val_count.get('Housing', 0)
    legal = val_count.get('Legal', 0)
    transportation = val_count.get('Transportation', 0)
    utilities = val_count.get('Utilities', 0)

    need_list = [adult_education,
                  child_related,
                  commodities,
                  employment,
                  financial,
                  food,
                  health,
                  housing,
                  legal,
                  transportation,
                  utilities,
                ]

    return need_list

def get_previous_cases(row, case_indexed_by_client, p_base_line_disconnect):
    """
    Apply to each row in cleaned cases
    """
    case_id = row['case_id']
    client_id = row['HL Client ID']

    current_case_closed_date = datetime.strptime(row['Case Date/Time Closed'], '%m/%d/%Y %I:%M %p')

    results = case_indexed_by_client.ix[[client_id]][['ID (Case Sensitive)', 'Case Date/Time Closed','binary_outcome']]
    results['c_closed_date'] = results.apply(lambda row: datetime.strptime(row['Case Date/Time Closed'], '%m/%d/%Y %I:%M %p') if pd.notnull(row['Case Date/Time Closed']) else np.nan, axis=1)

    closed_case_results = results[results['c_closed_date'].notnull()]
    previous_cases_sorted = closed_case_results[closed_case_results['c_closed_date']<current_case_closed_date].sort(['c_closed_date'])

    if len(previous_cases_sorted) == 0:
        previous_exist = 0
        p_disconnect_previous_case = p_base_line_disconnect
    else:
        previous_exist = 1
        p_disconnect_previous_case = previous_cases_sorted.iloc[-1]['binary_outcome']


    return p_disconnect_previous_case

def calculate_total_missed_contacts(row, activity_indexed_by_case, n_days=10):
    """
    Applied to cleaned_data, returns number of missed Advocate contacts for each case
    activity_indexed_by_case should be pre-sorted on case_id and activity completion date
    A missed contact is a contact that took place more than n_days since the last contact
    n_days is defaulted to 10 because Advocates are trained to contact within 10 days of the last contact
    """
    case_id = row['case_id']
    case_activities = activity_indexed_by_case.ix[[case_id]]
    # We only care about activities that are Progress Notes since
    # most contacts are recorded as that Type.
    case_progress_notes = case_activities[case_activities['Type'] == 'Progress Note']
    progress_notes_completed_dates = case_progress_notes['Converted Completed Date']
    num_missed = 0
    for ind in range(0, len(case_progress_notes) - 1): # len(...) - 1 so that every pairwise contact up to the last one can be computed
        previous_contact = progress_notes_completed_dates.iloc[ind]
        current_contact = progress_notes_completed_dates.iloc[ind + 1]
        delta = current_contact - previous_contact
        if delta.days > 10:
            num_missed += 1
    return num_missed

def calculate_total_contacts(row, activity_indexed_by_case, n_days=10):
    """
    Applied to cleaned_data, returns number of missed Advocate contacts for each case
    activity_indexed_by_case should be pre-sorted on case_id and activity completion date
    A missed contact is a contact that took place more than n_days since the last contact
    n_days is defaulted to 10 because Advocates are trained to contact within 10 days of the last contact
    """
    case_id = row['case_id']
    case_activities = activity_indexed_by_case.ix[[case_id]]
    # We only care about activities that are Progress Notes since
    # most contacts are recorded as that Type.
    case_progress_notes = case_activities[case_activities['Type'] == 'Progress Note']
    return len(case_progress_notes)

def get_client_referred(row):
    """
    Applied to a dataframe
    Returns whether a client was referred to HL
    """
    client_referral_origin = row['How Was Client Referred to HL?']
    referral =  [
                'Referred by non-clinical staff (medical asst., registration, legal services, etc)',
                'Referred by friend/relative',
                'Referred by health care provider (MD, RN, Social Work)',
                ]
    not_referral = [
                'No referral, advocate outreach brought client to HL',
                'No referral today, returning client',
                'Unknown',
                'No referral, marketing materials/signage brought client to HL',
                'Negative screen',
                ]
    if client_referral_origin in referral:
        return 1
    else:
        return 0
