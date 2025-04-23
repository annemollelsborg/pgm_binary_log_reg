# Load the data set employees_aatrition_dataset_10000.csv into a pandas dataframe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Set the style of seaborn
sns.set(style="whitegrid")

# Load the data set
data = pd.read_csv('employee_attrition_dataset_10000.csv')

# One hot encode the categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Attrition':
        dummies = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)
# Convert the target variable to a binary variable
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
# Convert the data types of the columns to float
data = data.astype(float)

# save the data set to a csv file
data.to_csv('employee_attrition_dataset_10000_processed.csv', index=False)

# Print the first 5 rows of the data set
print(data.head())
# Print the shape of the data set
print(data.shape)


# Make a correlation matrix and print save it to a file
def plot_correlation_matrix(data):
    # Calculate the correlation matrix
    corr = data.corr()
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    # Draw the heatmap without annotations
    sns.heatmap(corr, cmap='coolwarm', annot=False, square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.show()
# Plot the correlation matrix
plot_correlation_matrix(data)

# Plot the distribution of the target variable
def plot_target_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Attrition', data=data, palette='Set2')
    plt.title('Distribution of Attrition')
    plt.xlabel('Attrition')
    plt.ylabel('Count')
    plt.savefig('target_distribution.png')
    plt.show()
# Plot the target distribution
plot_target_distribution(data)