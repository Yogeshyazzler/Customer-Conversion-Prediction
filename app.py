
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from streamlit_option_menu import option_menu

with open('/content/sample_data/job_encoder.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract learned job categories
job_list = model.categories_[0]

def job_encoded(job):
    if job not in job_list:
        print(f"Warning: '{job}' is not in the trained job list")
        return np.zeros(len(job_list) - 1)  # Return a flat zero vector
    else:
        return model.transform([[job]]).flatten()

def marital_status_encoded(status):
    mapping = {'divorced': 0, 'single': 1, 'married': 2}
    return mapping.get(status, -1)

def education_encoded(edu):
    mapping = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
    return mapping.get(edu, -1)

def calls_encoded(call_type):
    mapping = {'unknown':0,'telephone':1, 'cellular':2}
    return mapping.get(call_type, -1)

with open('/content/sample_data/month_encoder.pkl', 'rb') as f:
    month_model = pickle.load(f)

# Fetch the trained month categories dynamically
month_list = month_model.categories_[0]

def month_encoded(mon):
    """Encodes a given month using the trained OneHotEncoder."""
    if mon not in month_list:
        print(f"Warning: '{mon}' is not in the trained month list. Encoding as all zeros.")
        return np.zeros(len(month_list) - 1)  # Return a flat zero vector
    else:
        return month_model.transform([[mon]]).flatten()

with open('/content/sample_data/prev_outcome_encoder.pkl', 'rb') as f:
    model = pickle.load(f)

# List of valid previous outcomes
valid_outcomes = ['unknown', 'failure', 'other', 'success']

label=LabelEncoder()
label.fit(valid_outcomes)

# Function to encode previous outcome
def encode_prev_outcome_encoded(outcome):
    if outcome not in valid_outcomes:
        raise ValueError(f"Invalid outcome: {outcome}. Choose from {valid_outcomes}")
    return label.transform([outcome])[0]


with open('/content/sample_data/logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

with open('/content/sample_data/decision_tree_model.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

with open('/content/sample_data/random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('/content/sample_data/knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('/content/sample_data/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

def logistic_prediction(age, job, marital, education_qual, call_type,dur,mon,day,num_calls, prev_outcome):
  pd_age = np.log1p(age)
  pd_job = job_encoded(job)
  pd_marital = marital_status_encoded(marital)
  pd_education = education_encoded(education_qual)
  pd_call_type = calls_encoded(call_type)
  pd_day=np.sqrt(day)
  pd_num_calls=np.log(num_calls)
  pd_month = month_encoded(mon)
  pd_duration = np.log1p(dur)
  pd_prev_outcome = encode_prev_outcome_encoded(prev_outcome)
  data = np.array([[pd_age, pd_job, pd_marital, pd_education, pd_call_type, pd_month,pd_day, pd_duration,pd_num_calls,pd_prev_outcome]])
  logistic_pred = logistic_model.predict(data)
  if logistic_pred==1:
    st.success("Yes")
    return logistic_pred
  else:
    st.error("No")
    return logistic_pred

def decision_tree(age, job, marital, education_qual, call_type,dur,mon,day,num_calls, prev_outcome):
  pd_age = np.log1p(age)
  pd_job = job_encoded(job)
  pd_marital = marital_status_encoded(marital)
  pd_education = education_encoded(education_qual)
  pd_call_type = calls_encoded(call_type)
  pd_month = month_encoded(mon)
  pd_duration = np.log1p(dur)
  pd_day=np.sqrt(day)
  pd_num_calls=np.log(num_calls)
  pd_prev_outcome = encode_prev_outcome_encoded(prev_outcome)
  # Combine all features into a single array
  data = np.hstack([pd_age, pd_job, pd_marital, pd_education, pd_call_type, pd_month, pd_duration, pd_day, pd_num_calls, pd_prev_outcome])
  data = data.reshape(1, -1)
  decision_pred = decision_tree_model.predict(data)
  if decision_pred==1:
    st.success("Yes")
    return decision_pred
  else:
    st.error("No")
    return decision_pred

def Random_forest_predict(age, job, marital, education_qual, call_type,dur,mon,day,num_calls, prev_outcome):
  pd_age = np.log1p(age)
  pd_job = job_encoded(job)
  pd_marital = marital_status_encoded(marital)
  pd_education = education_encoded(education_qual)
  pd_call_type = calls_encoded(call_type)
  pd_month = month_encoded(mon)
  pd_duration = np.log1p(dur)
  pd_day=np.sqrt(day)
  pd_num_calls=np.log(num_calls)
  pd_prev_outcome = encode_prev_outcome_encoded(prev_outcome)
  data = np.array([[pd_age, pd_job, pd_marital, pd_education, pd_call_type, pd_month, pd_duration,pd_day,pd_num_calls, pd_prev_outcome]])
  Random_forest_pred = random_forest_model.predict(data)
  if Random_forest_pred==1:
    st.success("Yes")
    return Random_forest_pred
  else:
    st.error("No")
    return Random_forest_pred

def knn_prediction(age, job, marital, education_qual, call_type,dur,mon,day,num_calls, prev_outcome):
  pd_age = np.log1p(age)
  pd_job = job_encoded(job)
  pd_marital = marital_status_encoded(marital)
  pd_education = education_encoded(education_qual)
  pd_call_type = calls_encoded(call_type)
  pd_month = month_encoded(mon)
  pd_duration = np.log1p(dur)
  pd_day=np.sqrt(day)
  pd_num_calls=np.log(num_calls)
  pd_prev_outcome = encode_prev_outcome_encoded(prev_outcome)
  data = np.array([[pd_age, pd_job, pd_marital, pd_education, pd_call_type, pd_month, pd_duration,pd_day,pd_num_calls, pd_prev_outcome]])
  knn_pred = knn_model.predict(data)
  if knn_pred==1:
    st.success("Yes")
    return knn_pred
  else:
    st.error("No")
    return knn_pred

def svm_prediction(age, job, marital, education_qual, call_type,dur,mon,day,num_calls, prev_outcome):
  pd_age = np.log1p(age)
  pd_job = job_encoded(job)
  pd_marital = marital_status_encoded(marital)
  pd_education = education_encoded(education_qual)
  pd_call_type = calls_encoded(call_type)
  pd_month = month_encoded(mon)
  pd_duration = np.log1p(dur)
  pd_day=np.sqrt(day)
  pd_num_calls=np.log(num_calls)
  pd_prev_outcome = encode_prev_outcome_encoded(prev_outcome)
  data = np.array([[pd_age, pd_job, pd_marital, pd_education, pd_call_type, pd_month, pd_duration,pd_day,pd_num_calls, pd_prev_outcome]])
  svm_pred = svm_model.predict(data)
  if svm_pred==1:
    st.success("Yes")
    return svm_pred
  else:
    st.error("No")
    return svm_pred


with st.sidebar:
    select = option_menu("Main Menu", ["Home", "Customer Prediction","Other Models prediction" ,"About"])
    st.sidebar.write("")

if select == "Home":
    st.header("Introduction:")
    st.write('''Customer Sales Prediction is a data-driven approach that helps businesses forecast future sales based on customer behavior, purchase history, and market trends. By leveraging machine learning and statistical models, companies can make informed decisions about inventory management, marketing strategies, and revenue growth.''')

    st.header("Key Factors Affecting Sales Prediction:")
    st.write('''1) Customer Demographics (Age, Gender, Location)''')
    st.write('''2) Purchase Behavior (Frequency, Order Value, Preferred Categories)''')
    st.write('''3) Seasonal Trends (Holiday Sales, Festive Discounts)''')
    st.write('''4) Marketing Impact (Discounts, Promotions, Ad Engagement)''')
    st.write('''5) Economic Conditions (Inflation, Competitor Pricing)''')

if select == "Customer Prediction":
  col1,col2 = st.columns(2)

  with col1:
    st.header("Customer Prediction")
    age = st.number_input("Enter Age", min_value=0, max_value=100)
    job = st.selectbox("Select the job", job_list.tolist())
    marital=st.selectbox("select marital_status",['married', 'single', 'divorced'])
    education_qual = st.selectbox("select Education qualification",['tertiary', 'secondary', 'unknown', 'primary'] )
    num_calls = st.number_input("enter number of calls",min_value=1)

  with col2:
    st.header("")
    call_type = st.selectbox("select calltype",['unknown', 'cellular', 'telephone'])
    mon =  st.selectbox("select Month",['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
    'mar', 'apr', 'sep'])
    dur = st.number_input("call duration",min_value=1)
    day=st.number_input("enter day",min_value=1,max_value=31)
    prev_outcome = st.selectbox("select Previous outcomes",['unknown', 'failure', 'other', 'success'])

    if st.button("predict"):

      prediction2 = decision_tree(age, job, marital, education_qual, call_type,dur,mon,day,num_calls, prev_outcome)
      st.write(f"The Decision Tree Classification predicted value = {prediction2}")
      st.write()

if select == "Other Models prediction":
  col1,col2 = st.columns(2)

  with col1:
    st.header("Customer Prediction")
    age = st.number_input("Enter Age", min_value=0, max_value=100)
    job = st.selectbox("Select the job", ['management', 'technician', 'entrepreneur', 'blue-collar',
          'unknown', 'retired', 'admin.', 'services', 'self-employed',
          'unemployed', 'housemaid', 'student'])
    marital=st.selectbox("select marital_status",['married', 'single', 'divorced'])
    education_qual = st.selectbox("select Education qualification",['tertiary', 'secondary', 'unknown', 'primary'] )
    num_calls = st.number_input("enter number of calls",min_value=1)

  with col2:
    st.header("")
    call_type = st.selectbox("select calltype",['unknown', 'cellular', 'telephone'])
    mon = st.selectbox("Select Month", month_list.tolist())
    dur = st.number_input("call duration",min_value=1)
    day=st.number_input("enter day",min_value=1,max_value=31)
    prev_outcome = st.selectbox("select Previous outcomes",['unknown', 'failure', 'other', 'success'])

    if st.button("predict"):

      prediction1 = logistic_prediction(age, job, marital, education_qual, call_type,dur,mon,day,num_calls, prev_outcome)
      st.write(f"The Logistic Classification predicted value = {prediction1}")
      st.write()

      prediction3 = Random_forest_predict(age, job, marital, education_qual, call_type,dur,mon,day,num_calls,  prev_outcome)
      st.write(f"The Random Forest Classification predicted value = {prediction3}")
      st.write()

      prediction4 = knn_prediction(age, job, marital, education_qual, call_type,dur,mon,day,num_calls,  prev_outcome)
      st.write(f"The KNN Classification predicted value = {prediction4}")
      st.write()

      prediction5 = svm_prediction(age, job, marital, education_qual, call_type,dur,mon,day,num_calls,  prev_outcome)
      st.write(f"The SVM Classification predicted value = {prediction5}")
      st.write()

if select == 'About':
  st.header("About")
  st.write(''' The data is from new-age insurance company and employ mutiple outreach plans to sell term insurance to your customers. Telephonic marketing campaigns still remain one of the most effective way to reach out to people however they incur a lot of cost. Hence, it is important to identify the customers that are most likely to convert beforehand so that they can be specifically targeted via call. The historical marketing data of the insurance company and are required to build a ML model that will predict if a client will subscribe to the insurance''')
