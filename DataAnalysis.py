import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset loading
def load_data():
    return pd.read_csv(r'C:\Users\Acer\Desktop\Video+Game+Sales\vgchartz-2024.csv')
    



# Data exploration
def explore_data(data):
    print("Dataset Info:")
    print(data.info())

    print("\nFirst 5 Rows of the Dataset:")
    print(data.head())

    print("\nSummary Statistics (for numeric columns):")
    print(data.describe())

# CHeck missing values
def handle_missing_values(data):
    print("\nMissing Values (Before Handling):")
    print(data.isnull().sum())

    # Fill numeric missing values 
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Fill categorical missing values
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    print("\nMissing Values (After Handling):")
    print(data.isnull().sum())

# Visualize numeric and categorical data
def visualize_data(data):
    data.select_dtypes(include=[np.number]).hist(bins=15, figsize=(12, 8), color='purple', edgecolor='black')
    plt.suptitle('Histograms for Numeric Columns')
    plt.tight_layout()
    plt.show()

    # Boxplot for numeric data
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data.select_dtypes(include=[np.number]), palette='pastel')
    plt.title('Box Plot of Numeric Columns')
    plt.xticks(rotation=45)
    plt.show()

    # Count plots for categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=col, palette="Set3")
        plt.title(f'Count Plot for {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Correlation analysis
def correlation_matrix(data):
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] == 0:
        print("\nNo numeric data available for correlation.")
        return

    corr = numeric_data.corr()
    print("\nCorrelation Matrix:")
    print(corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Correlation Heatmap')
    plt.show()

# Main function
def main():
    data = load_data()
    explore_data(data)
    handle_missing_values(data)
    visualize_data(data)
    correlation_matrix(data)

if __name__ == "__main__":
    main()
    main()
