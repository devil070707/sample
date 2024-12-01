import pandas as pd
import numpy as np

# Load the dataset
file_path = 'House Data.csv'

try:
    df = pd.read_csv(file_path)
    
    # Display basic dataset information
    print("Dataset Info:")
    print(df.info())
    print("\nDataset Preview:")
    print(df.head())

    # Define variables
    categorical_variable = 'district'
    quantitative_variable = 'price'

    # Check if the specified columns exist
    if categorical_variable in df.columns and quantitative_variable in df.columns:
        # Store original row count
        original_count = len(df)
        
        # Convert 'price' to numeric if it's not already
        if df[quantitative_variable].dtype == 'object':
            df[quantitative_variable] = pd.to_numeric(
                df[quantitative_variable].str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )
        
        # Drop rows with missing values in the specified columns
        df_clean = df.dropna(subset=[quantitative_variable, categorical_variable])
        rows_dropped = original_count - len(df_clean)
        if rows_dropped > 0:
            print(f"\nRemoved {rows_dropped} rows with missing values.")
        
        # Calculate grouped summary statistics
        grouped_stats = df_clean.groupby(categorical_variable)[quantitative_variable].agg(
            ['count', 'mean', 'median', 'min', 'max', 'std']
        ).round(2).reset_index()

        # Print summary statistics
        print(f"\nSummary statistics of '{quantitative_variable}' grouped by '{categorical_variable}':")
        print(grouped_stats)

        # Overall statistics for the quantitative variable
        print("\nOverall statistics of the quantitative variable:")
        print(df_clean[quantitative_variable].describe().round(2))

        # Visualization (Optional, enhances understanding)
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_clean, x=categorical_variable, y=quantitative_variable, palette='pastel')
        plt.title(f"Distribution of '{quantitative_variable}' by '{categorical_variable}'")
        plt.xlabel(categorical_variable.capitalize())
        plt.ylabel(quantitative_variable.capitalize())
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    else:
        # Error handling for missing columns
        missing_cols = [
            col for col in [categorical_variable, quantitative_variable] if col not in df.columns
        ]
        print(f"Error: Column(s) {', '.join(missing_cols)} not found in the dataset.")
        print(f"Available columns: {', '.join(df.columns)}")
        
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
