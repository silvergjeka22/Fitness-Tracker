# helper/tools.py

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import normaltest


class Tools:
    def __init__(self):
        pass

    def tune_dbscan(self, df, outliers_cols, eps_range=(0.1, 2.0, 0.1), min_samples_range=(5, 20)):
        """
        Function to tune DBSCAN parameters (eps and min_samples) for outlier detection
        and interpolate the outliers.

        Parameters:
        - df: The DataFrame containing the sensor data
        - outliers_cols: List of column names to apply DBSCAN
        - eps_range: Range of values for the `eps` parameter (default: (0.1, 2.0, 0.1))
        - min_samples_range: Range of values for the `min_samples` parameter (default: (5, 20))

        Returns:
        - best_eps: The best `eps` value found
        - best_min_samples: The best `min_samples` value found
        - df_scaled: Scaled DataFrame
        """
        # Standard scale the data
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[outliers_cols] = scaler.fit_transform(df[outliers_cols])
        
        best_eps = None
        best_min_samples = None
        min_outliers = float('inf')

        for eps in np.arange(eps_range[0], eps_range[1], eps_range[2]):
            for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
                # Apply DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                outlier_labels = dbscan.fit_predict(df_scaled[outliers_cols])

                # Count the number of outliers
                num_outliers = (outlier_labels == -1).sum()

                # If this combination produces fewer outliers, update the best parameters
                if num_outliers < min_outliers:
                    min_outliers = num_outliers
                    best_eps = eps
                    best_min_samples = min_samples

        # Display the results
        print(f"Best DBSCAN parameters: eps = {best_eps}, min_samples = {best_min_samples}")
        print(f"Number of outliers with these parameters: {min_outliers}")

        return best_eps, best_min_samples, df_scaled

    def check_normal_distribution(self, data, columns):
        """
        Function to check if columns of the DataFrame follow a normal distribution.
        Parameters:
        - data: The DataFrame
        - columns: List of columns to check normality for
        
        Prints the result for each column.
        """
        for col in columns:
            stat, p = normaltest(data[col])
            if p > 0.05:
                print(f'{col} is normally distributed (p-value: {p})')
            else:
                print(f'{col} is not normally distributed (p-value: {p})')
