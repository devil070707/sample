import pandas as pd
import numpy as np

# Read the CSV file
# Assuming the file is named 'iris.csv'
df = pd.read_csv('IRIS.csv')

# Group the data by species
for species in df['species'].unique():
    print(f"\nStatistical Analysis for {species}")
    print("=" * 50)
    
    # Filter data for current species
    species_data = df[df['species'] == species]
    
    # Calculate statistics for each numerical column
    numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for column in numeric_columns:
        print(f"\nStatistics for {column}:")
        print("-" * 30)
        
        # Calculate basic statistics
        stats = {
            'Mean': species_data[column].mean(),
            'Standard Deviation': species_data[column].std(),
            'Minimum': species_data[column].min(),
            '25th Percentile': species_data[column].quantile(0.25),
            'Median': species_data[column].median(),
            '75th Percentile': species_data[column].quantile(0.75),
            'Maximum': species_data[column].max()
        }
        
        # Print the statistics
        for stat_name, value in stats.items():
            print(f"{stat_name}: {value:.2f}")

# Get overall summary using pandas describe function
print("\nOverall Summary Statistics:")
print("=" * 50)
print(df.groupby('species').describe())