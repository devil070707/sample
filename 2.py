import pandas as pd

# Load the Telecom Churn dataset

data = pd.read_csv(r"C:\Users\ASUS\Desktop\DSML_practical\Telecom Churn.csv")


# Display the first few rows of the dataset for an overview
print("Dataset Overview:")
print(data.head())

# Display summary statistics using describe()
print("\nSummary Statistics:")
print(data.describe())

# Compute and display the minimum values for each column
print("\nMinimum values for each column:")
print(data.min(numeric_only=True))

# Compute and display the maximum values for each column
print("\nMaximum values for each column:")
print(data.max(numeric_only=True))

# Compute and display the mean values for each column
print("\nMean values for each column:")
print(data.mean(numeric_only=True))

# Compute Range (Max - Min) for each numeric column
numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
range_values = numeric_data.max() - numeric_data.min()
print("\nRange (Max - Min) for each column:")
print(range_values)

# Compute and display the standard deviation for each column
print("\nStandard deviation for each column:")
print(data.std(numeric_only=True))

# Compute and display the variance for each column
print("\nVariance for each column:")
print(data.var(numeric_only=True))

# Compute and display percentiles (25th, 50th, 75th) for each numeric column
percentiles = [0.25, 0.5, 0.75]  # Convert to fractions between 0 and 1
numeric_data = data.select_dtypes(include=['number'])  # Ensure only numeric columns are included
print("\nPercentiles (25th, 50th, 75th) for each numeric column:")
print(numeric_data.quantile(percentiles))