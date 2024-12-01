import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
file_path = r"Titanic.csv"
df = pd.read_csv(file_path)

# Set Seaborn theme
sns.set_theme(style="whitegrid")

# Exploratory Analysis with Multiple Visualizations
def analyze_titanic_patterns():
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))

    # 1. Survival by Passenger Class
    plt.subplot(2, 2, 1)
    sns.barplot(x='Pclass', y='Survived', data=df, palette='muted')
    plt.title('Survival Rate by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')

    # 2. Age Distribution by Survival
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Survived', y='Age', data=df, palette='pastel')
    plt.title('Age Distribution by Survival Status')
    plt.xlabel('Survived (0 = No, 1 = Yes)')
    plt.ylabel('Age')

    # 3. Survival by Gender
    plt.subplot(2, 2, 3)
    sns.barplot(x='Sex', y='Survived', data=df, palette='husl')
    plt.title('Survival Rate by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Survival Rate')

    # 4. Fare Distribution by Passenger Class
    plt.subplot(2, 2, 4)
    sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set3')
    plt.title('Fare Distribution by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Fare')

    plt.tight_layout()
    plt.show()

analyze_titanic_patterns()
