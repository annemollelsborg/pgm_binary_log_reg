# Load the employee_attrition_data.csv file and visualize the data. Make a historgram of each attribute

import pandas as pd
import matplotlib.pyplot as plt

# Load the data and one hot encode the categorical variables
data = pd.read_csv('employee_attrition_dataset_10000.csv')
# One hot encode the categorical variables to convert them into numerical format
data = pd.get_dummies(data)

# Print the header of the data frame to verify the one hot encoding
print(data.head())

# Create a histogram of each attribute
data.hist()
plt.show()

# Create a histogram of each attribute with a density plot
data.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
plt.show()

# Create a box plot of each attribute
data.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()

# Create a correlation matrix
correlations = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()