def get_nan_value(column_name, df):
    """
    Returns {column: (count_nans, percent_nans)}
    """
    column = df[column_name]
    count_nans = sum(column.isnull())
    total = len(column)
    percent_nans = float(count_nans)/total
    return {column_name: (total, count_nans, percent_nans)}
