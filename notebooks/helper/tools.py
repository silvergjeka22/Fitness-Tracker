from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import normaltest
from scipy.signal import butter, filtfilt, lfilter
from sklearn.decomposition import PCA
import copy
from sklearn.metrics import silhouette_score


############### Outlier detection ####################
class OutlierDetection:
    def __init__(self):
        pass

    def tune_dbscan(self, df, outliers_cols, eps_range=(0.1, 2.0, 0.1), min_samples_range=(5, 20)):
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[outliers_cols] = scaler.fit_transform(df[outliers_cols])
        
        best_eps = None
        best_min_samples = None
        best_score = -1  # Silhouette score

        total_steps = len(np.arange(eps_range[0], eps_range[1], eps_range[2])) * (min_samples_range[1] - min_samples_range[0] + 1)
        step = 1
        
        print(f"Starting DBSCAN tuning: {total_steps} total combinations to check...\n")
        for eps in np.arange(eps_range[0], eps_range[1], eps_range[2]):
            print(f"Checking eps = {eps:.2f}")
            for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
                print(f"  Step {step}/{total_steps}: eps={eps:.2f}, min_samples={min_samples}", end=" --> ")
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(df_scaled[outliers_cols])

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_outliers = (labels == -1).sum()
                n_points = len(labels)

                if n_clusters > 1:  # for >1 clusters
                    try:
                        score = silhouette_score(df_scaled[outliers_cols][labels != -1], labels[labels != -1])
                    except:
                        score = -1
                else:
                    score = -1

                print(f"Clusters: {n_clusters}, Outliers: {n_outliers}, Silhouette Score: {score:.4f}")

                # Optional: Only accept if outliers are less than 10%
                if score > best_score and (n_outliers / n_points) < 0.1:
                    print(f"    New best! Silhouette Score {score:.4f}")
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

                step += 1

        print(f"\nBest DBSCAN parameters: eps = {best_eps}, min_samples = {best_min_samples}")
        print(f"Best silhouette score: {best_score:.4f}")

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

###### Filtering functions ######
class LowPassFilter:
        def low_pass_filter(self, data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True):
            nyquist = 0.5 * sampling_frequency
            cutoff_norm = cutoff_frequency / nyquist
            b, a = butter(order, cutoff_norm, btype="low")
            filtered = filtfilt(b, a, data_table[col]) if phase_shift else lfilter(b, a, data_table[col])
            data_table[col + "_lowpass"] = filtered
            return data_table
        
##### Temporal abstraction functions ####
class MeanTemporalAbstraction:

    def abstract_mean(self, data_table, cols, window_size):
        for col in cols:
            new_col_name = f"{col}_temp_mean_ws_{window_size}"
            data_table[new_col_name] = data_table[col].rolling(window_size).mean()
        return data_table

