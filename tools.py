from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class Tools:
    def __init__(self):
        pass

    def tune_dbscan_and_interpolate(self, df, outliers_cols, eps_range=(0.1, 2.0, 0.1), min_samples_range=(5, 20)):
        """
        Function to tune DBSCAN parameters (eps and min_samples) for outlier detection.

        Parameters:
        - df: The DataFrame
        - outliers_cols: List of column names to apply DBSCAN
        - eps_range: Range of values for the `eps` parameter (default: (0.1, 2.0, 0.1))
        - min_samples_range: Range of values for the `min_samples` parameter (default: (5, 20))

        Returns:
        - best_eps: The best `eps`
        - best_min_samples: The best `min_samples`
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
