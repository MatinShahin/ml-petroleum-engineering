import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def detect_outliers_iqr(data, column_name, threshold=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method
    
    Parameters:
    -----------
    data : pandas DataFrame
        The input data
    column_name : str
        The column to analyze for outliers
    threshold : float, optional
        The threshold multiplier for IQR (default: 1.5)
    
    Returns:
    --------
    tuple
        (lower_bound, upper_bound, outlier_indices)
    """
    # Calculate Q1, Q3, and IQR
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    # Find outliers
    outlier_indices = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)].index
    
    return lower_bound, upper_bound, outlier_indices

def visualize_outliers(data, column_name, lower_bound, upper_bound, outlier_indices):
    """
    Visualize the data with outliers highlighted
    
    Parameters:
    -----------
    data : pandas DataFrame
        The input data
    column_name : str
        The column to visualize
    lower_bound : float
        The lower bound for outliers
    upper_bound : float
        The upper bound for outliers
    outlier_indices : pandas Index
        Indices of outlier points
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(range(len(data)), data[column_name], alpha=0.5, label='Normal points')
    plt.scatter(outlier_indices, data.loc[outlier_indices, column_name], 
                color='red', label='Outliers')
    
    # Add bounds
    plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower bound')
    plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper bound')
    
    plt.title(f'Outlier Detection for {column_name}')
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.legend()
    plt.grid(True)
    plt.show()

def remove_outliers(data, column_name, lower_bound, upper_bound):
    """
    Remove outliers from the data based on the given bounds
    
    Parameters:
    -----------
    data : pandas DataFrame
        The input data
    column_name : str
        The column to process
    lower_bound : float
        The lower bound for outliers
    upper_bound : float
        The upper bound for outliers
    
    Returns:
    --------
    pandas DataFrame
        Data with outliers removed
    """
    return data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]

def main():
    # Load your data
    df = pd.read_excel(r'C:\Users\ASUS\Desktop\ift2.xlsx', sheet_name='Sheet2')
    
    # Create a copy of the original data
    df_clean = df.copy()
    
    # Get all numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Analyze each numeric column
    for column in numeric_columns:
        print(f"\nAnalyzing column: {column}")
        lower_bound, upper_bound, outliers = detect_outliers_iqr(df, column)
        visualize_outliers(df, column, lower_bound, upper_bound, outliers)
        
        # Print summary
        print(f"Number of outliers detected: {len(outliers)}")
        print(f"Lower bound: {lower_bound:.2f}")
        print(f"Upper bound: {upper_bound:.2f}")
        print("\nOutlier values:")
        print(df.loc[outliers, column])
        print("-" * 50)
        
        # Remove outliers from the clean dataset
        df_clean = remove_outliers(df_clean, column, lower_bound, upper_bound)
    
    # Print summary of removed data
    print("\nSummary of data cleaning:")
    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows after removing outliers: {len(df_clean)}")
    print(f"Number of rows removed: {len(df) - len(df_clean)}")
    
    # Save the cleaned data to a new Excel file
    output_file = r'C:\Users\ASUS\Desktop\ift2_cleaned.xlsx'
    df_clean.to_excel(output_file, index=False)
    print(f"\nCleaned data saved to: {output_file}")

if __name__ == "__main__":
    main() 