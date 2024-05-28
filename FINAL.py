#!/usr/bin/env python
# coding: utf-8

# ## Group 26:
#     Minjia Li
#     Jay Mahn
#     Jiahui Liu
#     Lu Zeng

# # Train_data

# ## Data Cleaning

# ### CurrentTask, LastTaskCompleted-Encoding 

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.impute import KNNImputer

train_data_path = 'data/train_data.csv'
train_df = pd.read_csv(train_data_path)

label_encoder = LabelEncoder()

if train_df['CurrentTask'].dtype == 'object':
    train_df['CurrentTask'] = label_encoder.fit_transform(train_df['CurrentTask'].astype(str))

if train_df['LastTaskCompleted'].dtype == 'object':
    train_df['LastTaskCompleted'] = label_encoder.fit_transform(train_df['LastTaskCompleted'].astype(str))

output_file_path = 'train_data_knn_imputed.csv'
train_df.to_csv(output_file_path, index=False)


# ### CurrentTask, LastTaskCompleted-KNN

# In[2]:


import numpy as np

train_data = pd.read_csv('train_data_knn_imputed.csv')

# Select the two features 'CurrentTask_TargetEncoded' and 'LastTaskCompleted_TargetEncoded'
features_to_impute = ['CurrentTask', 'LastTaskCompleted']

# Initialize KNN imputer
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values
imputed_data = imputer.fit_transform(train_data[features_to_impute])

# Transfer the filled data back to the DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=features_to_impute, index=train_data.index)

# Re-assign the filled features back to the original dataset
train_data[features_to_impute] = imputed_df[features_to_impute]

print("Processed training data with imputed values:")
print(train_data.head())

# Save the imputed data to a new csv file
train_output_file_path = 'train_data_knn_imputed.csv'
train_data.to_csv(train_output_file_path, index=False)


# ### CurrentGameMode-Encoding

# In[3]:


from sklearn.preprocessing import LabelEncoder

# load the data
train_data = pd.read_csv('train_data_knn_imputed.csv')
print(train_data['CurrentGameMode'].unique())

# Initialize LabelEncoder
le = LabelEncoder()

# Encoding 'CurrentGameMode' using Label Encoding
train_data['CurrentGameMode'] = le.fit_transform(train_data['CurrentGameMode'])

train_data.head()


# ### CurrentGameMode-KNN

# In[4]:


# Initialize KNN Imputer
imputer = KNNImputer(n_neighbors=5)

# Select feature 'CurrentGameMode_LabelEncoded' to be imputed
features_to_impute = ['CurrentGameMode', 'CurrentTask']

# Impute the missing values
imputed_data = imputer.fit_transform(train_data[features_to_impute])

# Transfer the filled data back to the DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=features_to_impute, index=train_data.index)

# Re-new the original 'CurrentGameMode_LabelEncoded' data 
train_data['CurrentGameMode'] = imputed_df['CurrentGameMode']

print("Processed training data with imputed CurrentGameMode:")
print(train_data.head())

# Save and renew the file
train_output_file_path = 'train_data_knn_imputed.csv'
train_data.to_csv(train_output_file_path, index=False)


# ### TimeUtc-Convert Format

# In[5]:


train_data = pd.read_csv("train_data_knn_imputed.csv")

# Ensure TimeUtc is a datetime type
train_data['TimeUtc'] = pd.to_datetime(train_data['TimeUtc'])

# Extract components
train_data['Year'] = train_data['TimeUtc'].dt.year
train_data['Month'] = train_data['TimeUtc'].dt.month
train_data['Day'] = train_data['TimeUtc'].dt.day
train_data['Hour'] = train_data['TimeUtc'].dt.hour
train_data['Minute'] = train_data['TimeUtc'].dt.minute
train_data['Second'] = train_data['TimeUtc'].dt.second

# Separate weekday and weekend days
train_data['Weekday'] = train_data['TimeUtc'].dt.dayofweek
train_data['WeekendFlag'] = (train_data['TimeUtc'].dt.weekday >= 5).astype(int)

# Categorize different times of day into periods
time_bins = [0, 6, 12, 18, 24]  # Define time_bins as 0-6, 6-12, 12-18, 18-24
time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
train_data['PeriodOfDay'] = pd.cut(train_data['TimeUtc'].dt.hour, bins=time_bins, labels=time_labels, right=False)

# Remove the original Timestamp column to avoid redundancy
train_data.drop('TimeUtc', axis=1, inplace=True)

train_data.head()


# ### TimeUtc-Encoding

# In[6]:


#encoding for TimeOfDay.
train_data = pd.get_dummies(train_data, columns=['PeriodOfDay'], drop_first=False)

train_data.head()


# In[7]:


# Save the transformation and encoding for 'TimeUtc' to the 'train_data_knn_imputed.csv' 
train_output_file_path = 'train_data_knn_imputed.csv'
train_data.to_csv(train_output_file_path, index=False)


# ### LevelProgressionAmount-KNN

# In[8]:


import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Read the CSV file
file_path = 'train_data_knn_imputed.csv'
df = pd.read_csv(file_path)

# Select the column to encode and impute
column_to_encode_and_impute = 'LevelProgressionAmount'

# Encode the column
label_encoder = LabelEncoder()
df[column_to_encode_and_impute] = label_encoder.fit_transform(df[column_to_encode_and_impute].astype(str))

# Create KNNImputer instance
imputer = KNNImputer(n_neighbors=5)

# Perform KNN imputation on the selected column
df[[column_to_encode_and_impute]] = imputer.fit_transform(df[[column_to_encode_and_impute]])

# Standardize the imputed column
scaler = StandardScaler()
df[[column_to_encode_and_impute]] = scaler.fit_transform(df[[column_to_encode_and_impute]])

# Save the modified data back to the original file
df.to_csv(file_path, index=False)

print(f'Successfully encoded and performed KNN imputation on {column_to_encode_and_impute}. The result has been saved back to the original file {file_path}')


# ### QuestionTiming-Dummy

# In[9]:


import pandas as pd

file_path = 'train_data_knn_imputed.csv'
df = pd.read_csv(file_path)

df_encoded = pd.get_dummies(df, columns=['QuestionTiming'])

df_encoded.to_csv(file_path, index=False)

print(f'{file_path}')


# ### Mapping UserID with ResponseValue

# In[10]:


file_path = 'train_data_knn_imputed.csv'
df = pd.read_csv(file_path)

# average responsevalue per user
user_means = df.groupby('UserID')['ResponseValue'].mean()

# overall average responsevalue
global_mean = df['ResponseValue'].mean()

# Creating new feature using average responsevalue per user
df['UserAvgResponse'] = df['UserID'].map(user_means)

# imputing missing value
df['UserAvgResponse'].fillna(global_mean, inplace=True)

# save new feature to file
df.to_csv(file_path, index=False)

print(f'Successfully updated UserAvgResponse and saved the data back to the file {file_path}')


# ### Userid-Label

# In[11]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_data_path = 'train_data_knn_imputed.csv'
train_df = pd.read_csv(train_data_path)

label_encoder = LabelEncoder()

train_df['UserID'] = label_encoder.fit_transform(train_df['UserID'])

output_file_path = 'train_data_knn_imputed.csv'
train_df.to_csv(output_file_path, index=False)

print(train_df.head())


# ## Feature Importance

# #### CurrentGameMode

# In[12]:


train_data = pd.read_csv('train_data_knn_imputed.csv')
# Fit the model
model = ols('ResponseValue ~ C(CurrentGameMode)', data=train_data).fit()

# Perform ANOVA
anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame

# Print the ANOVA table
print(anova_results)


# #### QuestionTiming

# In[13]:


from scipy.stats import pointbiserialr

train_data = pd.read_csv('train_data_knn_imputed.csv')

# Calculate point-biserial correlation
correlation, p_value = pointbiserialr(train_data['QuestionTiming_System Initiated'], train_data['ResponseValue'])
print(f"Correlation: {correlation}, P-value: {p_value}")


# In[14]:


from scipy.stats import pointbiserialr

train_data = pd.read_csv('train_data_knn_imputed.csv')

# Calculate point-biserial correlation
correlation, p_value = pointbiserialr(train_data['QuestionTiming_User Initiated'], train_data['ResponseValue'])
print(f"Correlation: {correlation}, P-value: {p_value}")


# #### CurrentSessionLength

# In[15]:


correlation = train_data['CurrentSessionLength'].corr(train_data['ResponseValue'])
print(f'Correlation between CurrentSessionLength and ResponseValue: {correlation}')


# #### CurrentTask & LastTaskCompleted

# In[16]:


train_data = pd.read_csv('train_data_knn_imputed.csv')

corr_current_task = train_data['CurrentTask'].corr(train_data['ResponseValue'])
corr_last_task_completed = train_data['LastTaskCompleted'].corr(train_data['ResponseValue'])

print(f'Correlation between CurrentTask and ResponseValue: {corr_current_task}')
print(f'Correlation between LastTaskCompleted and ResponseValue: {corr_last_task_completed}')


# #### LevelProgressionAmount

# In[17]:


correlation = train_data['LevelProgressionAmount'].corr(train_data['ResponseValue'])
print(f'Correlation between LevelProgressionAmount and ResponseValue: {correlation}')


# #### TimeUtc

# ##### Weekday

# In[18]:


weekday_unique_values = train_data['Weekday'].unique()
print(weekday_unique_values)

from scipy.stats import f_oneway

grouped_data = [train_data[train_data['Weekday'] == i]['ResponseValue'] for i in range(7)]

f_stat, p_val = f_oneway(*grouped_data)
print('F-statistic:', f_stat)
print('P-value:', p_val)


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(x='Weekday', y='ResponseValue', data=train_data)
plt.title('ResponseValue by Weekday')
plt.xlabel('Weekday')
plt.ylabel('ResponseValue')
plt.show()


# In[20]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=train_data['ResponseValue'], groups=train_data['Weekday'], alpha=0.05)
print(tukey)

tukey.plot_simultaneous()
plt.show()


# ##### Weekend

# In[21]:


from scipy.stats import pointbiserialr, ttest_ind

WeekendFlag_unique_values = train_data['WeekendFlag'].unique()
print(WeekendFlag_unique_values)

correlation, p_value = pointbiserialr(train_data['WeekendFlag'], train_data['ResponseValue'])
print(f"Point-Biserial Correlation: {correlation}, P-value: {p_value}")

mean_weekday = train_data[train_data['WeekendFlag'] == 0]['ResponseValue'].mean()
mean_weekend = train_data[train_data['WeekendFlag'] == 1]['ResponseValue'].mean()
print(f"Mean on Weekdays: {mean_weekday}, Mean on Weekends: {mean_weekend}")

group_weekday = train_data[train_data['WeekendFlag'] == 0]['ResponseValue']
group_weekend = train_data[train_data['WeekendFlag'] == 1]['ResponseValue']
t_stat, p_val = ttest_ind(group_weekday, group_weekend)
print(f"T-statistic: {t_stat}, P-value: {p_val}")

plt.figure(figsize=(8, 6))
sns.boxplot(x='WeekendFlag', y='ResponseValue', data=train_data)
plt.title('ResponseValue by WeekendFlag')
plt.xlabel('WeekendFlag')
plt.ylabel('ResponseValue')
plt.show()


# ##### Period of day

# In[22]:


import statsmodels.formula.api as smf

formula = 'ResponseValue ~ PeriodOfDay_Afternoon + PeriodOfDay_Evening + PeriodOfDay_Night'

model = smf.ols(formula, data=train_data).fit()

print(model.summary())


# #### Lasso

# In[23]:


import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = 'train_data_knn_imputed.csv'  
final_data = pd.read_csv(file_path)

target = 'ResponseValue'
features = final_data.drop(columns=[target,'QuestionType'])
target_data = final_data[target]

X_train, X_test, y_train, y_test = train_test_split(features, target_data, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)  
lasso.fit(X_train, y_train)

coefficients = lasso.coef_

feature_importance = pd.DataFrame({'Feature': features.columns, 'Importance': coefficients})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Lasso Regression')
plt.show()


# #### XGBoost

# In[24]:


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

file_path = 'train_data_knn_imputed.csv' 
final_data = pd.read_csv(file_path)

target = 'ResponseValue'
features = final_data.drop(columns=[target,'QuestionType'])
target_data = final_data[target]

X_train, X_test, y_train, y_test = train_test_split(features, target_data, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)

importance = xgb_model.feature_importances_

feature_importance_xgb = pd.DataFrame({'Feature': features.columns, 'Importance': importance})
feature_importance_xgb = feature_importance_xgb.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_xgb['Feature'], feature_importance_xgb['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from XGBoost')
plt.show()


# # Test_data

# In[27]:


test_data= pd.read_csv('data/test_data.csv')


# ### CurrentTask, LastTaskCompleted-Encoding 

# In[28]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

test_data_path = 'data/test_data.csv'
test_df = pd.read_csv(test_data_path)

label_encoder = LabelEncoder()

if test_df['CurrentTask'].dtype == 'object':
    test_df['CurrentTask'] = label_encoder.fit_transform(test_df['CurrentTask'].astype(str))

if test_df['LastTaskCompleted'].dtype == 'object':
    test_df['LastTaskCompleted'] = label_encoder.fit_transform(test_df['LastTaskCompleted'].astype(str))

output_file_path = 'test_data_knn_imputed.csv'
test_df.to_csv(output_file_path, index=False)


# ### CurrentTask, LastTaskCompleted-KNN

# In[29]:


from sklearn.impute import KNNImputer

test_data = pd.read_csv('test_data_knn_imputed.csv')

# Select the two features 'CurrentTask_TargetEncoded' and 'LastTaskCompleted_TargetEncoded'
features_to_impute = ['CurrentTask', 'LastTaskCompleted']

# Initialize KNN imputer
imputer = KNNImputer(n_neighbors=5)

# Impute the missing values
imputed_data = imputer.fit_transform(test_data[features_to_impute])

# Transfer the filled data back to the DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=features_to_impute, index=test_data.index)

# Re-assign the filled features back to the original dataset
test_data[features_to_impute] = imputed_df[features_to_impute]

print("Processed testing data with imputed values:")
print(test_data.head())

# Save the imputed data to a new csv file
test_output_file_path = 'test_data_knn_imputed.csv'
test_data.to_csv(test_output_file_path, index=False)


# ### CurrentGameMode-Encoding

# In[30]:


from sklearn.preprocessing import LabelEncoder

# load the data
test_data = pd.read_csv('test_data_knn_imputed.csv')
print(test_data['CurrentGameMode'].unique())

# Initialize LabelEncoder
le = LabelEncoder()

# Encoding 'CurrentGameMode' using Label Encoding
test_data['CurrentGameMode'] = le.fit_transform(test_data['CurrentGameMode'])


# ### CurrentGameMode-KNN

# In[31]:


# Initialize KNN Imputer
imputer = KNNImputer(n_neighbors=5)

# Select feature 'CurrentGameMode_LabelEncoded' to be imputed
features_to_impute = ['CurrentGameMode', 'CurrentTask']

# Impute the missing values
imputed_data = imputer.fit_transform(test_data[features_to_impute])

# Transfer the filled data back to the DataFrame
imputed_df = pd.DataFrame(imputed_data, columns=features_to_impute, index=test_data.index)

# Re-new the original 'CurrentGameMode_LabelEncoded' data 
test_data['CurrentGameMode'] = imputed_df['CurrentGameMode']

print("Processed testing data with imputed CurrentGameMode:")
print(test_data.head())

# Save and renew the file
test_output_file_path = 'test_data_knn_imputed.csv'
test_data.to_csv(test_output_file_path, index=False)


# ### TimeUtc-Convert Format

# In[32]:


test_data = pd.read_csv("test_data_knn_imputed.csv")

# Ensure TimeUtc is a datetime type
test_data['TimeUtc'] = pd.to_datetime(test_data['TimeUtc'])

# Extract components
test_data['Year'] = test_data['TimeUtc'].dt.year
test_data['Month'] = test_data['TimeUtc'].dt.month
test_data['Day'] = test_data['TimeUtc'].dt.day
test_data['Hour'] = test_data['TimeUtc'].dt.hour
test_data['Minute'] = test_data['TimeUtc'].dt.minute
test_data['Second'] = test_data['TimeUtc'].dt.second

# Separate weekday and weekend days
test_data['Weekday'] = test_data['TimeUtc'].dt.dayofweek
test_data['WeekendFlag'] = (test_data['TimeUtc'].dt.weekday >= 5).astype(int)

# Categorize different times of day into periods
time_bins = [0, 6, 12, 18, 24]  # Define time_bins as 0-6, 6-12, 12-18, 18-24
time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
test_data['PeriodOfDay'] = pd.cut(test_data['TimeUtc'].dt.hour, bins=time_bins, labels=time_labels, right=False)

# Remove the original Timestamp column to avoid redundancy
test_data.drop('TimeUtc', axis=1, inplace=True)


# ### TimeUtc-Encoding

# In[33]:


#encoding for TimeOfDay.
test_data = pd.get_dummies(test_data, columns=['PeriodOfDay'], drop_first=False)

test_data.head()


# In[34]:


# Save the transformation and encoding for 'TimeUtc' to the 'train_data_knn_imputed.csv' 
test_output_file_path = 'test_data_knn_imputed.csv'
test_data.to_csv(test_output_file_path, index=False)


# ### LevelProgressionAmount-KNN

# In[35]:


import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Read the CSV file
file_path = 'test_data_knn_imputed.csv'
df = pd.read_csv(file_path)

# Select the column to encode and impute
column_to_encode_and_impute = 'LevelProgressionAmount'

# Encode the column
label_encoder = LabelEncoder()
df[column_to_encode_and_impute] = label_encoder.fit_transform(df[column_to_encode_and_impute].astype(str))

# Create KNNImputer instance
imputer = KNNImputer(n_neighbors=5)

# Perform KNN imputation on the selected column
df[[column_to_encode_and_impute]] = imputer.fit_transform(df[[column_to_encode_and_impute]])

# Standardize the imputed column
scaler = StandardScaler()
df[[column_to_encode_and_impute]] = scaler.fit_transform(df[[column_to_encode_and_impute]])

# Save the modified data back to the original file
df.to_csv(file_path, index=False)

print(f'Successfully encoded and performed KNN imputation on {column_to_encode_and_impute}. The result has been saved back to the original file {file_path}')


# ### QuestionTiming-Dummy

# In[36]:


import pandas as pd

file_path = 'test_data_knn_imputed.csv'
df = pd.read_csv(file_path)

df_encoded = pd.get_dummies(df, columns=['QuestionTiming'])

df_encoded.to_csv(file_path, index=False)

print(f'{file_path}')


# ### Adding UserAvgResponse

# In[37]:


import pandas as pd

train_data_path = 'data/train_data.csv'
train_df = pd.read_csv(train_data_path)

user_means = train_df.groupby('UserID')['ResponseValue'].mean()

global_mean = train_df['ResponseValue'].mean()

test_data_path = 'test_data_knn_imputed.csv'
test_df = pd.read_csv(test_data_path)

test_df['UserAvgResponse'] = test_df['UserID'].map(user_means)

test_df['UserAvgResponse'].fillna(global_mean, inplace=True)

test_df.to_csv(test_data_path, index=False)

print(f'successfully add {test_data_path}')


# # Model

# ### Hyperparameter Tuning

# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import lightgbm as lgb
import joblib

# Load the data
data_path = 'train_data_knn_imputed.csv'
data = pd.read_csv(data_path)

# Define features and target variable
features = [
    "CurrentSessionLength", "CurrentGameMode", "CurrentTask",
    "LastTaskCompleted", "LevelProgressionAmount", "Month", "Year",
    "WeekendFlag", "PeriodOfDay_Night", "QuestionTiming_System Initiated", "UserAvgResponse"
]
X = data[features]
y = data['ResponseValue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter search space
search_spaces = {
    'num_leaves': Integer(20, 50),
    'max_depth': Integer(5, 50),
    'learning_rate': Real(0.01, 0.3, 'log-uniform'),
    'n_estimators': Integer(100, 500),
    'min_data_in_leaf': Integer(1, 20),
    'feature_fraction': Real(0.7, 1.0),
    'bagging_fraction': Real(0.7, 1.0),
    'bagging_freq': Integer(1, 10),
    'min_split_gain': Real(0.0, 20.0)
}

# Initialize the LightGBM regressor
lgb_regressor = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', metric='rmse', random_state=42, force_row_wise=True)

# Perform Bayesian optimization
opt = BayesSearchCV(lgb_regressor, search_spaces, n_iter=32, n_jobs=-1, cv=3, scoring='neg_mean_squared_error', random_state=42)

# Fit the model
opt.fit(X_train, y_train)

# Output the best parameters
print("Best parameters:", opt.best_params_)


# ### LightGBM

# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb

# Load data
data_path = 'train_data_knn_imputed.csv'
data = pd.read_csv(data_path)

# Replace spaces in feature names with underscores
data.columns = data.columns.str.replace(' ', '_')

# Define features and target variable
features = [
    "CurrentSessionLength", "CurrentGameMode", "CurrentTask",
    "LastTaskCompleted", "LevelProgressionAmount", "Month", "Year",
    "WeekendFlag", "PeriodOfDay_Night", "QuestionTiming_System_Initiated", "UserAvgResponse"
]
X = data[features]
y = data['ResponseValue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare datasets for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set model parameters based on optimization results
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l2', 'mae'],
    'num_leaves': 44,  
    'learning_rate': 0.04124065492949547,
    'feature_fraction': 0.7355768413737933,
    'bagging_fraction': 0.9261701294787221,
    'bagging_freq': 10,
    'verbose': 0,
    'max_depth': 14,  
    'min_data_in_leaf': 20,  
    'min_split_gain': 0.43114217732353133,
    'n_estimators': 382
}

# Number of boosting rounds
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], callbacks=[lgb.early_stopping(stopping_rounds=10)])

# Predict and evaluate the model
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Output the evaluation metrics
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
print(f"Mean Absolute Error: {mae}")


# In[38]:


import pandas as pd
import lightgbm as lgb


test_data_path = 'test_data_knn_imputed.csv'  
test_data = pd.read_csv(test_data_path)

features = [
    "CurrentSessionLength", "CurrentGameMode", "CurrentTask",
    "LastTaskCompleted", "LevelProgressionAmount", "Month", "Year",
    "WeekendFlag", "PeriodOfDay_Night", "QuestionTiming_System Initiated", "UserAvgResponse"
]
X_test = test_data[features]

y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

predicted_data = pd.DataFrame(y_pred)

output_file_path = 'predicted.csv'
predicted_data.to_csv(output_file_path, index=False, header=False)

print(f"Predicted data saved to: {output_file_path}")


# In[ ]:




