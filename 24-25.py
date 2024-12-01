import pandas as pd
import numpy as np

# Load the dataset
file_path = 'Lung Cancer.csv'
try:
    data = pd.read_csv(file_path)
    
    print("\n=== Dataset Loaded Successfully ===")
    print("Dataset shape:", data.shape)
    print("\nPreview of the dataset:")
    print(data.head())
    
    # =====================================
    # Assignment 24: Basic Data Operations
    # =====================================

    # 24.1: Counting unique values in each column
    print("\n=== Count of Unique Values in Each Column ===")
    for column in data.columns:
        print(f"{column}: {data[column].nunique()} unique values")
    
    # 24.2: Checking format/data type of each column
    print("\n=== Data Types of Each Column ===")
    print(data.dtypes)
    
    # 24.3 & 24.5: Identifying and filling missing values
    print("\n=== Missing Values Count ===")
    missing_values = data.isnull().sum()
    print(missing_values)
    print("\nPercentage of missing values:")
    print((missing_values / len(data) * 100).round(2))
    
    # Fill numeric missing values with median
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Fill categorical missing values with mode
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # 24.4: Converting variable data types
    data['Age'] = data['Age'].astype('int32')
    for col in numeric_columns:
        if col != 'Age':
            data[col] = pd.to_numeric(data[col], downcast='integer')
    
    # =====================================
    # Assignment 25: Data Cleaning and Transformation
    # =====================================
    
    # 25.1: Data Cleaning
    # Remove duplicates
    data = data.drop_duplicates()
    print("\n=== Duplicates Removed ===")
    print("Dataset shape after removing duplicates:", data.shape)
    
    # Handle outliers using IQR
    for col in numeric_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)
    
    # 25.2: Data Transformation
    # Normalize numeric columns
    def normalize_column(column):
        return (column - column.min()) / (column.max() - column.min())
    
    for col in numeric_columns:
        data[f'{col}_normalized'] = normalize_column(data[col])
    
    # Derived features
    data['is_smoker'] = (data['Smokes'] > 0).astype(int)
    data['age_group'] = pd.cut(data['Age'], bins=[0, 30, 50, 70, 100], 
                               labels=['Young', 'Middle', 'Senior', 'Elderly'])
    data['risk_score'] = (
        normalize_column(data['Smokes']) * 0.3 +
        normalize_column(data['Alkhol']) * 0.2 +
        normalize_column(data['Age']) * 0.5
    )
    
    # Final Quality Check
    print("\n=== Final Data Quality Check ===")
    print("Missing values after cleaning:")
    print(data.isnull().sum())
    print("\nDataset shape after transformations:", data.shape)
    print("\nData types after transformation:")
    print(data.dtypes)
    
    # Save the cleaned and transformed data
    output_file = 'LungCancer_cleaned_transformed_v1.csv'
    data.to_csv(output_file, index=False)
    print(f"\nCleaned and transformed dataset saved as '{output_file}'")
    
    # Display sample of the final dataset
    print("\n=== Sample of Final Transformed Dataset ===")
    print(data.head())
    
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
