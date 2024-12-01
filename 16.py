import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
file_path = r"Titanic.csv"
df = pd.read_csv(file_path)

# Ensure 'Fare' column exists and handle missing values if any
if 'Fare' in df.columns:
    # Drop missing values in 'Fare' column
    fare_data = df['Fare'].dropna()

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Create a histogram with KDE overlay
    sns.histplot(data=fare_data, bins=50, kde=True, color='skyblue', edgecolor='black')

    # Calculate statistics
    mean_fare = fare_data.mean()
    median_fare = fare_data.median()
    max_fare = fare_data.max()
    min_fare = fare_data.min()

    # Add vertical lines for statistics
    plt.axvline(mean_fare, color='red', linestyle='--', label=f'Mean Fare: £{mean_fare:.2f}')
    plt.axvline(median_fare, color='green', linestyle='-', label=f'Median Fare: £{median_fare:.2f}')
    plt.axvline(max_fare, color='purple', linestyle='-.', label=f'Max Fare: £{max_fare:.2f}')
    plt.axvline(min_fare, color='orange', linestyle=':', label=f'Min Fare: £{min_fare:.2f}')

    # Enhancing the plot
    plt.title('Distribution of Ticket Fares on Titanic', fontsize=16)
    plt.xlabel('Fare (in pounds)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', alpha=0.75)

    # Display the plot
    plt.show()

    # Additional Insights
    print("Fare Statistics:")
    print(f"- Mean Fare: £{mean_fare:.2f}")
    print(f"- Median Fare: £{median_fare:.2f}")
    print(f"- Maximum Fare: £{max_fare:.2f}")
    print(f"- Minimum Fare: £{min_fare:.2f}")

    print("\nFare Insights:")
    print("- A significant proportion of passengers paid low fares, as evident from the histogram.")
    print("- High fares likely correspond to first-class passengers.")
    print("- Outliers with very high fares could indicate luxurious accommodations.")
else:
    print("'Fare' column not found in the dataset.")
