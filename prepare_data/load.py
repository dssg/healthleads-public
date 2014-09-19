import DAL
import clean
import os

cleaned_file_name = os.environ.get('DATA_FILENAME', None)

raw_data_filepath = os.environ.get('RAW_DATA', None) 
model_data_filepath = os.environ.get('MODEL_DATA', None)

# the out file is a made-up directory

dal = DAL.DAL(raw_data_filepath, 'clean/')

# Loading the raw data
client = dal.load_csv('Clients and Dependents', {}, clean = False)
case = dal.load_csv('Cases', {}, clean = False)
need = dal.load_csv('Presenting Needs', {}, clean = False)
need = clean.clean_need(need)
activity = dal.load_csv('Activities', {}, clean = False)
advocate = dal.load_csv('Advocates', {}, clean = False)
location = dal.load_csv('Desk Locations', {}, clean = False)
connection = dal.load_csv('Resource Connections - Successful Non-ETO', {}, clean = False)
advocate_demographics = dal.load_csv('Advocate Demographics', {}, clean = False)
# Loading the cleaned data
merged_access_layer = DAL.DAL(model_data_filepath, '/clean/')
cleaned_data = merged_access_layer.load_csv(cleaned_file_name, {}, clean = False)

