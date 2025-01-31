
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("cleaned dataset.csv")

# Feature Engineering

# 1. Calculate Age where missing
def calculate_age(dob):
    if pd.isna(dob):
        return np.nan  # Handle missing DOBs
    try:
        today = date.today()
        born = datetime.strptime(dob, "%m/%d/%Y").date()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except ValueError:  # Handle invalid date formats
        return np.nan

df['Age'] = df.apply(lambda x: calculate_age(x['date_of_birth']) if pd.isna(x['Age']) else x['Age'], axis=1)

#2. Extract datetime features
for col in ['learner_signup_datetime', 'opportunity_end_date', 'entry_created_at', 'apply_date','opportunity_start_date']:
    try:
        df[col] = pd.to_datetime(df[col])  #Convert string representation to datetime objects
        df[col+'_year'] = df[col].dt.year
        df[col+'_month'] = df[col].dt.month
        df[col+'_day'] = df[col].dt.day
    except: 
        print(f"Skipping column {col} as conversion is not possible")

# 3. Time differences (days) - fill errors with a large negative value for easier filtering.
def safe_days_diff(date1, date2):
    try:
        return abs((date1 - date2).days)
    except TypeError:  # For NaT or unknown types
        return -9999

df['days_to_opp_end'] = df.apply(lambda row: safe_days_diff(row['learner_signup_datetime'], row['opportunity_end_date']), axis=1)
df['signup_to_apply'] = df.apply(lambda row: safe_days_diff(row['learner_signup_datetime'], row['apply_date']), axis=1)
df['signup_to_entry'] = df.apply(lambda row: safe_days_diff(row['learner_signup_datetime'], row['entry_created_at']), axis=1)


# 4. Encoding categorical features
df['gender'] = df['gender'].replace("Don't want to specify", "Other") # Replace "Don't want to specify" with "Other"
for col in ['opportunity_category','gender','country','current/intended_major','status_description','Status Description']: # List the categorical columns
    dummies = pd.get_dummies(df[col], prefix=col)  # Creates dummy variables
    df = pd.concat([df, dummies], axis=1) # Adds the dummy variables to the dataframe

# 5. Clean Institution and Major Names (Basic)
for col in ['institution_name', 'Institution Name', 'current/intended_major', 'Current/Intended Major']:
    df[col] = df[col].str.lower().str.replace('[^a-zA-Z0-9\s]', '').str.strip()  # Lowercase, remove special characters, strip whitespace



# Visualizations and Analysis (Examples)

# Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'].dropna(), kde=True) # Ignore NaN values for plotting.
plt.title('Age Distribution of Learners')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# Signup Trend over Time
plt.figure(figsize=(10, 6))
df['learner_signup_datetime'].value_counts().sort_index().plot()
plt.title('Learner Signup Trend')
plt.xlabel('Signup Date')
plt.ylabel('Number of Signups')
plt.show()

# Status Description counts
plt.figure(figsize=(8, 6))
df['status_description'].value_counts().plot(kind='bar')
plt.title('Status Description Counts')
plt.xlabel('Status Description')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right') # Rotate labels for better visibility
plt.tight_layout() # Adjusts layout to prevent labels from overlapping
plt.show()

# Country Distribution
plt.figure(figsize=(10, 6))
df['country'].value_counts().plot(kind='bar')
plt.title('Distribution of Learners by Country')
plt.xlabel('Country')
plt.ylabel('Number of Learners')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.tight_layout() # Adjusts subplot parameters for a tight layout
plt.show()



# Correlation Heatmap (for numeric features)
numeric_df = df.select_dtypes(include=np.number) # Subset for the numeric data
corr = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numeric Features')
plt.show()



#  Example: Analyzing Major vs. Status
plt.figure(figsize=(12, 8))
sns.countplot(y='current/intended_major', hue='status_description', data=df, order=df['current/intended_major'].value_counts().index)
plt.title('Current/Intended Major vs. Status Description')
plt.xlabel('Count')
plt.ylabel('Current/Intended Major')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# Save the engineered dataset (optional)
df.to_csv("engineered_dataset.csv", index=False)


print("Feature engineering and visualization complete.")