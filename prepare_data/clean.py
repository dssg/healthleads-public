def make_binary_outcome(outcome):
    adj_outcome = 1 if outcome == 4 else 0
    return adj_outcome

def clean_data(df):
	df.loc[1230, 'Gender'] = 'Female'
	df.loc[445, "Client Origin"] = "Screen dropped off & added to triage"
	return df

def clean_need(df):
    df['Need Desk Location'].replace(to_replace = 'NY Pres - WHFHC', value = 'NY Pres - WHFHC (Disabled)', inplace=True)
    return df
