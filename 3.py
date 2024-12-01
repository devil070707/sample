import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data into a pandas DataFrame (assuming your data is stored in a CSV file)
file_path = r"C:\Users\ASUS\Desktop\DSML_practical\House Data.csv"  # Path to the CSV file
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    data = None

if data is not None:
    # List of columns to clean (adjust according to your dataset)
    columns_to_clean = [
        'price', 'GrossSquareMeters', 'BuildingAge', 'NumberFloorsofBuilding', 
        'NumberOfBathrooms', 'NumberOfWCs', 'NetSquareMeters', 'RentalIncome', 
        'HallSquareMeters', 'WCSquareMeters', 'BathroomSquareMeters', 
        'BalconySquareMeters'
    ]

    # Clean each column
    for column in columns_to_clean:
        if column in data.columns:
            # Remove non-numeric characters (like $ and commas) using regex
            data[column] = data[column].replace({'\$': '', ',': ''}, regex=True)
            
            # Convert the column to numeric, forcing errors to NaN (Not a Number)
            data[column] = pd.to_numeric(data[column], errors='coerce')

            # Fill NaN values with the mean or other strategies (can be customized)
            if data[column].isna().sum() > 0:
                if data[column].dtype == 'object':
                    # For non-numeric columns, use the mode or custom handling
                    data[column] = data[column].fillna(data[column].mode()[0])
                else:
                    # Use the mean for numeric columns
                    data[column] = data[column].fillna(data[column].mean())

    # Calculate and print the statistics for each column
    for column in columns_to_clean:
        if column in data.columns:
            print(f"--- {column} ---")
            # Calculate statistics only if the column is numeric
            if pd.api.types.is_numeric_dtype(data[column]):
                # Print basic statistics: standard deviation, variance, percentiles
                print(f"Standard Deviation: {data[column].std():.2f}")
                print(f"Variance: {data[column].var():.2f}")
                print(f"25th Percentile: {data[column].quantile(0.25):.2f}")
                print(f"50th Percentile (Median): {data[column].quantile(0.50):.2f}")
                print(f"75th Percentile: {data[column].quantile(0.75):.2f}")
            else:
                print(f"Column '{column}' is not numeric. Skipping...")
            print()  # Blank line between columns

    # Optionally, you can also plot histograms for the cleaned numeric columns
    for column in columns_to_clean:
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column]) and not data[column].isna().all():
            plt.figure(figsize=(8, 6))
            plt.hist(data[column], bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

    # Example of checking if the data cleaned properly
    print(data.head())  # Print the first few rows to verify the changes
else:
    print("No data available for analysis.")




# Options for errors Parameter:
# errors='raise' (default):

# If a value cannot be converted, the function will raise an error and stop execution.
# errors='ignore':

# If a value cannot be converted, the function leaves it unchanged in its original form (no conversion occurs).
# errors='coerce':

# If a value cannot be converted to a number, it is replaced with NaN (Not a Number).