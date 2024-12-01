import pandas as pd
import numpy as np

# Load the dataset (replace with the actual file path if needed)
file_path = r"C:\Users\ASUS\Desktop\DSML_practical\Lipstick.csv"  # Ensure this path is correct
df = pd.read_csv(file_path)

# Display the dataset to understand its structure
print("Dataset:")
print(df.head())

# Encode categorical data into numerical values manually (if required)
# Adjusting for actual column names in the dataset
df['Age'] = df['Age'].map({'<21': 0, '21-35': 1, '>35': 2})  # Encoding for Age (assuming values like '<21', '21-35', '>35')
df['Income'] = df['Income'].map({'Low': 0, 'Medium': 1, 'High': 2})  # Encoding for Income
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})  # Encoding for Gender (assuming Male = 0, Female = 1)
df['Ms'] = df['Ms'].map({'Single': 0, 'Married': 1})  # Encoding for Marital Status
df['Buys'] = df['Buys'].map({'No': 0, 'Yes': 1})  # Encoding for Buys (target variable)

# Display the dataset after encoding
print("\nEncoded Dataset:")
print(df.head())

# Function to calculate entropy of a dataset
def entropy(data):
    total = len(data)  # Total number of examples
    pos_count = np.sum(data)  # Count of positive class (Buys = 1)
    neg_count = total - pos_count  # Count of negative class (Buys = 0)
    
    # If all examples are of one class, entropy is 0
    if pos_count == 0 or neg_count == 0:
        return 0
    
    # Calculate probabilities
    p_pos = pos_count / total
    p_neg = neg_count / total
    
    # Calculate entropy
    return -p_pos * np.log2(p_pos) - p_neg * np.log2(p_neg)

# Function to calculate information gain for a given feature
def information_gain(data, feature, target):
    unique_values = data[feature].unique()  # Get the unique values of the feature
    total_entropy = entropy(data[target])  # Calculate the entropy of the whole dataset
    
    # Calculate the weighted sum of entropies of subsets
    weighted_entropy = 0
    for value in unique_values:
        subset = data[data[feature] == value]  # Subset of data for this feature value
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])  # Weighted entropy for the subset
    
    # Information Gain is the difference between total entropy and weighted entropy
    return total_entropy - weighted_entropy

# Calculate the information gain for each feature
features = ['Age', 'Income', 'Gender', 'Ms']  # List of features (you can add more features here)
for feature in features:
    gain = information_gain(df, feature, 'Buys')
    print(f"Information Gain for '{feature}': {gain}")

# Find the feature with the highest information gain (root node)
best_feature = None
max_gain = -1

for feature in features:
    gain = information_gain(df, feature, 'Buys')
    if gain > max_gain:
        max_gain = gain
        best_feature = feature

print(f"\nRoot Node of the Decision Tree: {best_feature} with Information Gain: {max_gain}")