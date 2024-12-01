import pandas as pd

#pip install pandas , pip install openpyxl , import pandas as pd  print(pd.__version__)

#for csv pd.raed_csv and for excel pd.read_excel

# 1. upload dataset 
# file_path = r'C:\Users\91830\Downloads\Titanic.csv'
data = pd.read_csv(r"C:\Users\ASUS\Desktop\DSML_practical\Titanic.csv")

# 2. Display the first few rows of the dataset
print("First 5 rows:")
print(data.head())

# 3. Indexing and Selecting Data
print("\nSelecting the first 3 rows and specific columns (Name, Age):")
print(data.loc[:4, ['Name', 'Age']])  # Adjust columns based on your dataset structure

# 4. Sorting Data by 'Age'
sorted_data = data.sort_values(by='Age', ascending=True)
print("\nData sorted by Age:")
print(sorted_data.head())

# 5. Describe Attributes of Data
print("\nStatistical Summary:")
print(data.describe())

# 6. Check Data Types
print("\nData Types of Columns:")
print(data.dtypes)