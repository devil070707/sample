import pandas as pd
import numpy as np
from math import log2

# Step 1: Create the dataset
data = {
    'Age': ['Young', 'young', 'Middle', 'Old', 'Old', 'Old', 'Middle', 'Young', 
            'Young', 'Old', 'Young', 'Middle', 'Middle', 'Old'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 
               'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Married': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 
                'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'Health': ['Fair', 'Good', 'Fair', 'Fair', 'Fair', 'Good', 'Good', 'Fair', 
               'Fair', 'Fair', 'Good', 'Good', 'Fair', 'Good'],
    'Class': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
              'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Create frequency table for Age
print("\nStep 2: Frequency Table for Age")
age_freq = df['Age'].value_counts()
print(age_freq)

# Step 3: Calculate entropy of the target variable (Class)
def entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -sum(p * log2(p) for p in probabilities)

system_entropy = entropy(df['Class'])
print("\nStep 3: System Entropy")
print(f"System Entropy = {system_entropy:.4f}")

# Step 4: Calculate conditional entropy for each Age value
print("\nStep 4: Conditional Entropy for each Age value")
age_entropy = {}
weighted_entropy = 0

for age_value in df['Age'].unique():
    subset = df[df['Age'] == age_value]['Class']
    age_entropy[age_value] = entropy(subset)
    weight = len(subset) / len(df)
    weighted_entropy += weight * age_entropy[age_value]
    print(f"Entropy for Age = {age_value}: {age_entropy[age_value]:.4f}")

# Step 5: Calculate Information Gain
information_gain = system_entropy - weighted_entropy

print("\nStep 5: Final Calculations")
print(f"Weighted Entropy = {weighted_entropy:.4f}")
print(f"Information Gain = {information_gain:.4f}")

# Step 6: Display detailed split information
print("\nStep 6: Detailed Split Information")
for age_value in df['Age'].unique():
    subset = df[df['Age'] == age_value]
    class_dist = subset['Class'].value_counts()
    print(f"\nAge = {age_value}")
    print(f"Total instances: {len(subset)}")
    print("Class distribution:")
    print(class_dist)