from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import normaltest
from scipy.signal import butter, filtfilt, lfilter
from sklearn.decomposition import PCA
import copy


############### Outlier detection ####################
class OutlierDetection:
    def __init__(self):
        pass

    def tune_dbscan(self, df, outliers_cols, eps_range=(0.1, 2.0, 0.1), min_samples_range=(5, 20)):
        # Standard scale the data
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[outliers_cols] = scaler.fit_transform(df[outliers_cols])
        
        best_eps = None
        best_min_samples = None
        min_outliers = float('inf')
        
        # it starts form 0.1 to 2.0 with 0.1 step
        for eps in np.arange(eps_range[0], eps_range[1], eps_range[2]): 
            # it starts from 5 to 20
            for min_samples in range(min_samples_range[0], min_samples_range[1] + 1): # +1 to include the last value
                # Apply DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                outlier_labels = dbscan.fit_predict(df_scaled[outliers_cols])

                # the number of outliers
                num_outliers = (outlier_labels == -1).sum()

                if num_outliers < min_outliers:
                    min_outliers = num_outliers
                    best_eps = eps
                    best_min_samples = min_samples

        # Display the results
        print(f"Best DBSCAN parameters: eps = {best_eps}, min_samples = {best_min_samples}")
        print(f"Number of outliers with these parameters: {min_outliers}")

        return best_eps, best_min_samples, df_scaled

    def check_normal_distribution(self, data, columns):
        for col in columns:
            stat, p = normaltest(data[col])
            if p > 0.15: # 0.15 -> 85% normal distribution
                print(f'{col} is normally distributed (p-value: {p})')
            else:
                print(f'{col} is not normally distributed (p-value: {p})')

############### Feature engineering ####################

class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):

        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            # formula max-min normalization -> (x - mean) / (max - min)
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
            )
        return dt_norm

    def pca_variance(self, data_table, cols):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_ # The explained variance of each component.

    def apply_pca(self, data_table, cols, number_comp):

        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table