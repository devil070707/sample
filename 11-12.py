# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
# Make sure to put the correct path to your CSV file
iris_data = pd.read_csv('IRIS.csv')

# Assignment 11
print("Assignment 11:")
print("\n1. Features and their types:")
print(iris_data.dtypes)

# Create histogram for each feature
plt.figure(figsize=(12, 8))

# Sepal Length
plt.subplot(2, 2, 1)
plt.hist(iris_data['sepal_length'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')

# Sepal Width
plt.subplot(2, 2, 2)
plt.hist(iris_data['sepal_width'], bins=20, color='green', alpha=0.7)
plt.title('Histogram of Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')

# Petal Length
plt.subplot(2, 2, 3)
plt.hist(iris_data['petal_length'], bins=20, color='red', alpha=0.7)
plt.title('Histogram of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')

# Petal Width
plt.subplot(2, 2, 4)
plt.hist(iris_data['petal_width'], bins=20, color='purple', alpha=0.7)
plt.title('Histogram of Petal Width')
plt.xlabel('Petal Width')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Assignment 12
# Create box plots
plt.figure(figsize=(12, 6))

# Create box plot for all features
plt.subplot(1, 2, 1)
iris_data.boxplot(column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.title('Box Plot of Iris Features')
plt.ylabel('Value')

# Create individual box plots with seaborn for better visualization
plt.subplot(1, 2, 2)
sns.boxplot(data=iris_data.melt(value_vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']), 
            x='variable', y='value')
plt.title('Box Plot of Iris Features (Seaborn)')
plt.xticks(rotation=45)
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Calculate and print outliers
print("\nOutliers Analysis:")
for column in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    Q1 = iris_data[column].quantile(0.25)
    Q3 = iris_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = iris_data[(iris_data[column] < lower_bound) | (iris_data[column] > upper_bound)][column]
    
    print(f"\n{column}:")
    print(f"Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print("Outlier values:")
        print(outliers.values)

# Print basic statistical information
print("\nBasic Statistical Information:")
print(iris_data.describe())