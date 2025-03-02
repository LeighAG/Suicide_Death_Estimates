#python -m venv venv
#.\venv\Scripts\activate
import pandas as pd
import sqlite3
#import tkinter
import missingno as msno
#Missingno is a Python library specifically designed to help you visualize and understand missing data patterns in your datasets
import matplotlib.pyplot as plt
import seaborn as sns
import re


#import data
df_raw = pd.read_csv('data_suicide_rates.csv')
#6390 entries, 13 columns
#there are 5484 Estimates, and 6390 other rows

# create database
conn = sqlite3.connect('Suicide_rates.db')

df_raw.to_sql('raw_data', conn, if_exists='replace', index=False)

#1. Explore and understand the main dataset******************************************************************

#filter out Deaths per 100,000 resident population, crude and keep Deaths per 100,000 resident population, age-adjusted
df = pd.read_sql_query("SELECT * FROM raw_data WHERE UNIT_NUM != 2", conn)

# Completeness Check for missing data
completeness = df.isnull().sum() / len(df)

# Duplicates Check
duplicates = df.duplicated().sum()

# label_mapping = df[['STUB_LABEL', 'STUB_LABEL_NUM']].drop_duplicates().sort_values(by='STUB_LABEL_NUM')

# # Display the mapping as a table
# print("Show label mapping tabel", label_mapping)

# Group by the specified columns and calculate missing percentages in `ESTIMATE`
missing_summary = df.groupby(['STUB_LABEL_NUM', 'YEAR'])['ESTIMATE'].apply(lambda group: group.isnull().mean())

# Reset the index
missing_summary = missing_summary.reset_index()

missing_summary['STUB_LABEL_NUM'] = missing_summary['STUB_LABEL_NUM'].astype(str)
# Plot the missing data
plt.figure(figsize=(10, 6))
sns.barplot(
    data=missing_summary, 
    x='YEAR', 
    y='ESTIMATE', 
    hue='STUB_LABEL_NUM'
)

plt.title("Percentage of Missing Data in ESTIMATE by YEAR and STUB_LABEL_NUM")
plt.ylabel("Percentage Missing")
plt.xlabel("YEAR")
plt.legend(title="STUB_LABEL_NUM", bbox_to_anchor=(1.05, 1), loc='upper left')
# Rotate x-axis labels for better readability
plt.xticks(rotation=90)  # Rotate x-axis labels to vertical
plt.show()

#tables for missing data

missing_summary_between = missing_summary[
    (missing_summary['ESTIMATE'] > 0) & 
    (missing_summary['YEAR'].between(1985 , 1998))
]

# Ensure STUB_LABEL_NUM is treated as a string (or category) for clear labels
missing_summary_between['STUB_LABEL_NUM'] = missing_summary_between['STUB_LABEL_NUM'].astype(str)

# Display the resulting data as a table
print("Percentage of Missing Data in ESTIMATE between 1985 and by STUB_LABEL_NUM:")
print(missing_summary_between.to_string(index=False))  # Avoid printing the index column


#look closer to see missing data in 2018
missing_summary = missing_summary[
    (missing_summary['ESTIMATE'] > 0) & 
    (missing_summary['YEAR'] == 2018)
]

# Ensure STUB_LABEL_NUM is treated as a string (or category) for clear labels
missing_summary['STUB_LABEL_NUM'] = missing_summary['STUB_LABEL_NUM'].astype(str)

# Display the resulting data as a table
print("Percentage of Missing Data in ESTIMATE for 2018 by STUB_LABEL_NUM:")
print(missing_summary.to_string(index=False))  # Avoid printing the index column

#This shows significant lack of sex and race and ethnicity prior to 1985. 
#From 1985, there is missing data of two groups.  One missing in 2018, stub_label_num = 6.27.

#upon further investigation of visually understanding the dataset, I found that this was only added in 2018 (no other years) and they were blank.

#now I would like to create a cleaned dataset with 2000-2018, no 6.27 stub label and no unit that is crude (only want adj for age)

# Ensure YEAR is numeric
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')

#2. Create, explore and filter dataset to use for the project*************************************************************************

#create dataset with just stub label estimate, and year. 

query = """
SELECT STUB_LABEL, STUB_LABEL_NUM, YEAR, YEAR_NUM, ESTIMATE
FROM raw_data
WHERE UNIT_NUM = 1 AND YEAR >= 2000
"""

# Read the filtered data into a DataFrame
df_filtered = pd.read_sql_query(query, conn)

#create another column titled Sex for male, female and both
def classify_sex(label):
    label_lower = label.lower()  # Convert to lowercase for case-insensitive search
    if re.search(r'\bmale\b', label_lower):  # Match whole word 'male'
        return 1
    elif re.search(r'\bfemale\b', label_lower):  # Match whole word 'female'
        return 2
    else:
        return 3

# Apply the function to create the new 'Sex' column
df_filtered['Sex'] = df_filtered['STUB_LABEL'].apply(classify_sex)

# Display unique values in categorical columns
categorical_columns = ['STUB_LABEL', 'YEAR']  # Modify if other columns are categorical
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(df_filtered[col].unique())

# Count of unique values in each column
# print("\nUnique Value Counts:")
# print(df_filtered.nunique())

# Check for duplicate rows
# print("\nNumber of Duplicate Rows:")
# print(df_filtered.duplicated().sum())

# Identify duplicate rows
duplicate_rows = df_filtered[df_filtered.duplicated(keep=False)]  # keep=False shows all duplicates

# Identify rows where ESTIMATE is missing
missing_estimate_rows = df_filtered[df_filtered['ESTIMATE'].isna()]
#there are 5 duplicate rows and one missing ESTIMATE
# Display results
# print("Duplicate Rows:")
# print(duplicate_rows)

# print("\nRows with Missing ESTIMATE:")
# print(missing_estimate_rows)

#STILL NEED TO ELIMINATE STUB LABEL 6.27
df_filtered = df_filtered[df_filtered['STUB_LABEL_NUM'] != 6.27]
#NEED TO CORRECT DUPLICATE ROWS
df_filtered = df_filtered.drop_duplicates(keep='first')
 
# Sort by 'YEAR' to ensure the years are in order
df_filtered = df_filtered.sort_values(by=['STUB_LABEL_NUM', 'YEAR'])

# Calculate the change in 'ESTIMATE' from the previous year and create new column titled Estimate_change
df_filtered['Estimate_change'] = df_filtered.groupby('STUB_LABEL_NUM')['ESTIMATE'].diff()

# Display the updated DataFrame
# print(df_filtered[['STUB_LABEL_NUM', 'YEAR', 'ESTIMATE', 'Estimate_change']])

# Define the years to include (2000-2018)
required_years = set(range(2000, 2019))

# Group by 'STUB_LABEL_NUM' and verify that I am filtering those that contain all years from 2000 to 2018
df_filtered = df_filtered.groupby('STUB_LABEL_NUM').filter(
    lambda x: set(x['YEAR']) == required_years
)

# Display the first few rows to verify
print(df_filtered.head())

# Display basic information about the dataset
print("Dataset Info:")
print(df_filtered.info())

# Show the first few rows to preview the data
print("\nFirst 5 Rows:")
print(df_filtered.head())

# Get summary statistics for numerical columns
print("\nSummary Statistics:")
print(df_filtered.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for better aesthetics
sns.set_style("whitegrid")

# Create the line plot
plt.figure(figsize=(12, 7))

# Define labels for the Stub_Label_Num categories
label_mapping = {0.00: 'All Persons', 2.10: 'Males', 2.20: 'Females'}

# Loop through each Stub_Label_Num category and plot a line
for label_num, label_name in label_mapping.items():
    subset = df_filtered[df_filtered['STUB_LABEL_NUM'] == label_num]
    plt.plot(subset['YEAR'], subset['ESTIMATE'], label=label_name)

# Add titles and axis labels
plt.title('Estimate Over Years by Category (Stub_Label_Num)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Estimate', fontsize=12)

# Add a legend
plt.legend(title='Category')

# Display the plot
plt.show()

# Filter dataset for only STUB_LABELs that contain "FEMALE" (case insensitive)
female_df = df_filtered[df_filtered['STUB_LABEL'].str.contains("FEMALE", case=False, na=False)]

# Define a color palette for distinction
palette = sns.color_palette("coolwarm", len(female_df['STUB_LABEL_NUM'].unique()))

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.lineplot(
    data=female_df, 
    x='YEAR', 
    y='ESTIMATE', 
    hue='STUB_LABEL',  # Hue by STUB_LABEL instead of STUB_LABEL_NUM for clarity
    marker='o', 
    linewidth=2,
    palette=palette
)

# Customize the legend
plt.legend(title="STUB_LABEL (FEMALE Categories)", bbox_to_anchor=(1.05, 1), loc='upper left')

# Customize the graph
plt.title("ESTIMATE Over Time for STUB_LABELs Containing 'FEMALE'")
plt.xlabel("Year")
plt.ylabel("ESTIMATE")
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(True)

# Show the plot
plt.show()

# Filter dataset for only STUB_LABELs that contain "FEMALE" (case insensitive)
male_df = df_filtered[
    df_filtered['STUB_LABEL'].str.contains("MALE", case=False, na=False) & 
    ~df_filtered['STUB_LABEL'].str.contains("FEMALE", case=False, na=False)
]

# Define a color palette for distinction
palette = sns.color_palette("coolwarm", len(male_df['STUB_LABEL_NUM'].unique()))

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.lineplot(
    data=male_df, 
    x='YEAR', 
    y='ESTIMATE', 
    hue='STUB_LABEL',  # Hue by STUB_LABEL instead of STUB_LABEL_NUM for clarity
    marker='o', 
    linewidth=2,
    palette=palette
)

# Customize the legend
plt.legend(title="STUB_LABEL (MALE Categories)", bbox_to_anchor=(1.05, 1), loc='upper left')

# Customize the graph
plt.title("ESTIMATE Over Time for STUB_LABELs Containing 'MALE'")
plt.xlabel("Year")
plt.ylabel("ESTIMATE")
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(True)

# Show the plot
plt.show()


# Export DataFrame to Excel file
df_filtered.to_excel('filtered_data.xlsx', index=False)

print("Data exported successfully to filtered_data.xlsx") 

# Create column to remove the first 'Estimate_change' value for each 'STUB_LABEL_NUM' since there is no previous number to compare
df_filtered['Estimate_change_no_first'] = df_filtered.groupby('STUB_LABEL')['Estimate_change'].transform(lambda x: x.shift(-1))

# Group by 'STUB_LABEL_NUM' and calculate the average of 'Estimate_change_no_first' (excluding first row)
df_avg_estimate_change = df_filtered.groupby('STUB_LABEL')['Estimate_change_no_first'].mean().reset_index()

# Display the result
print(df_avg_estimate_change)

# Set the style for better aesthetics
sns.set_style("whitegrid")

# Create the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='STUB_LABEL', y='Estimate_change_no_first', data=df_avg_estimate_change, color='skyblue')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Add labels and title
plt.xlabel('STUB_LABEL', fontsize=12)
plt.ylabel('Average Estimate Change (Excluding First Year)', fontsize=12)
plt.title('Average Estimate Change by STUB_LABEL', fontsize=16)

# Display the plot
plt.tight_layout()
plt.show()


#export datasheet showing only Stub_Label and estimate change over time from 2000-2018
df_avg_estimate_change.to_excel('avg_estimate_change.xlsx', index=False)




