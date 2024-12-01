import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('Covid Vaccine Statewise.csv')

# Assignment 13
print("Assignment 13:")
print("\na. Dataset Description:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Group by state and calculate first dose and second dose
state_wise_vac = df.groupby('State')[['First Dose Administered', 'Second Dose Administered']].sum()

# Sort values in descending order for better visualization
state_wise_first = state_wise_vac['First Dose Administered'].sort_values(ascending=False)
state_wise_second = state_wise_vac['Second Dose Administered'].sort_values(ascending=False)

print("\nb. Number of persons state wise vaccinated for first dose:")
print(state_wise_first)

print("\nc. Number of persons state wise vaccinated for second dose:")
print(state_wise_second)

# Plotting state-wise vaccination data
plt.figure(figsize=(15, 6))
state_wise_first.plot(kind='bar')
plt.title('State-wise First Dose Vaccination')
plt.xlabel('State')
plt.ylabel('Number of People')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
state_wise_second.plot(kind='bar')
plt.title('State-wise Second Dose Vaccination')
plt.xlabel('State')
plt.ylabel('Number of People')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Assignment 14
print("\nAssignment 14:")
print("\nA. Dataset Description:")
print("\nShape of dataset:", df.shape)
print("\nColumns in dataset:", df.columns.tolist())
print("\nFirst few rows of dataset:")
print(df.head())

# Calculate total males and females vaccinated
total_males = df['Male(Individuals Vaccinated)'].sum()
total_females = df['Female(Individuals Vaccinated)'].sum()

print("\nB. Number of Males vaccinated:", total_males)
print("C. Number of Females vaccinated:", total_females)

# Create a pie chart for gender distribution
plt.figure(figsize=(10, 8))
plt.pie([total_males, total_females],
        labels=['Males', 'Females'],
        autopct='%1.1f%%',
        colors=['lightblue', 'lightpink'])
plt.title('Distribution of Vaccinated Population by Gender')
plt.show()

# Create a bar plot for gender comparison
plt.figure(figsize=(10, 6))
plt.bar(['Males', 'Females'], [total_males, total_females],
        color=['lightblue', 'lightpink'])
plt.title('Comparison of Vaccinated Males and Females')
plt.ylabel('Number of People Vaccinated')
plt.grid(True, axis='y')
plt.show()


plt.figure(figsize=(15, 6))

plt.title('Daily Vaccination Trend by Gender')
plt.xlabel('Date')
plt.ylabel('Number of People')
plt.grid(True)
plt.legend(['Males', 'Females'])
plt.tight_layout()
plt.show()