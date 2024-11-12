# Import necessary libraries for data manipulation and visualization
import numpy as np              # For numerical operations
import matplotlib.pyplot as plt # For plotting
import pandas as pd             # For data handling (DataFrames)
from scipy import stats         # For statistical functions
import seaborn as sns           # For advanced visualizations

# Import clustering and outlier detection tools from scikit-learn
from sklearn.cluster import DBSCAN          # DBSCAN clustering algorithm
from sklearn.preprocessing import StandardScaler # To standardize data
from sklearn.cluster import KMeans          # KMeans clustering algorithm
from sklearn.neighbors import LocalOutlierFactor # Tool for detecting outliers
# Sample data generation
np.random.seed(42)  # Set a random seed for reproducibility (ensures the same results every time)
# Generate 1000 normal data points with mean 0 and standard deviation 1
data_normal = np.random.normal(0, 1, 1000)
# Generate 50 outlier data points with mean 0 and standard deviation 5 (larger spread)
data_outliers = np.random.normal(0, 5, 50)

# Combine the normal data and outliers into one dataset
data = np.concatenate([data_normal, data_outliers])

# Create a DataFrame from the data, with a column named 'value'
df = pd.DataFrame(data, columns=['value'])

# Z-Score Method to detect outliers
# Calculate the Z-Score for each value in the 'value' column
df['zscore'] = np.abs(stats.zscore(df['value']))  # Absolute value of Z-Score

# Set a threshold for outlier detection (commonly 3)
# If the Z-Score is greater than 3, it's considered an outlier
outliers_zscore = df[df['zscore'] > 3]

# Print the outliers detected using Z-Score
print("Outliers detected using Z-Score:")
print(outliers_zscore)